"""12장 CNN 학습 코드를 분리 방식과 무관하게 다시 쓰기 위한 모듈 (12-4 사후 실험용).

12장 Notebook(12_wafermap_cnn)의 모델·학습·평가 코드와 같은 내용입니다.
학습·검증 인덱스를 인자로 받도록 바꿔, 검증 분리 방식을 바꿔 가며 같은 학습을 반복할 수 있게 했습니다.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .wafer_utils import CLASSES, classification_report_dict


def to_input(m):
    """uint8 맵(0·1·2) → 원-핫 3채널 float. contiguous()로 일반 메모리 배치로 바꾼다(12-1)."""
    return F.one_hot(m.long(), 3).permute(0, 3, 1, 2).float().contiguous()


def augment(x):
    """배치를 8개 묶음으로 나눠 묶음마다 다른 90도 회전·뒤집기를 적용한다."""
    out = torch.empty_like(x)
    perm = torch.randperm(len(x), device=x.device)
    for g, part in enumerate(perm.chunk(8)):
        t = torch.rot90(x[part], g % 4, dims=(2, 3))
        out[part] = torch.flip(t, dims=(3,)) if g >= 4 else t
    return out


class SmallCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def make_resnet(pretrained: bool):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


@torch.no_grad()
def predict(model, maps_gpu, idx, batch=512):
    model.eval()
    logits = [model(to_input(maps_gpu[idx[i:i + batch]])) for i in range(0, len(idx), batch)]
    return torch.cat(logits).softmax(1)


def class_weight(y_train: np.ndarray, device) -> torch.Tensor:
    """드문 유형일수록 큰 가중치(역제곱근). 12장과 같은 식."""
    counts = np.bincount(y_train, minlength=len(CLASSES))
    w = counts.astype(float) ** -0.5
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def train_model(make, seed, maps_gpu, y_gpu, y_all, idx_tr, idx_va, class_weight=None,
                lr=1e-3, epochs=30, patience=6, batch=256):
    """12장과 같은 학습: AdamW, 증강, 검증 macro F1 기준 조기 종료(patience 6)."""
    device = maps_gpu.device
    va_np = idx_va.cpu().numpy()
    torch.manual_seed(seed); np.random.seed(seed)
    model = make().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    best, best_state, wait, hist = -1, None, 0, []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = idx_tr[torch.randperm(len(idx_tr), device=device)]
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = loss_fn(model(augment(to_input(maps_gpu[b]))), y_gpu[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        pv = predict(model, maps_gpu, idx_va).argmax(1).cpu().numpy()
        f1 = classification_report_dict(y_all[va_np], pv)["macro_f1"]
        hist.append({"epoch": ep + 1, "train_loss": round(total / len(perm), 4), "val_macro_f1": f1})
        if f1 > best:
            best, best_state, wait = f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, hist, sec, best
