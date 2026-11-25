"""PoC 3-B. MVTec AD 이상 탐지 공용 코드 (책 13장).

MVTec AD는 CC BY-NC-SA 4.0 라이선스입니다. 비상업적 목적으로만 쓰고, 데이터 파일을 다시 배포하지 않습니다.
출처: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SEED = 42
DATA_ROOT = Path("data/mvtec_ad")
WORK_DIR = Path("work")                       # Anomalib이 만드는 임시 파일 (저장소에 올리지 않음)
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
FOCUS = ["bottle", "screw", "carpet"]          # 본문에서 자세히 보는 카테고리 (결과를 보기 전에 정함)
IMAGE_SIZE = 256


# ---------------------------------------------------------------- 데이터 구조
def dataset_summary(root: str | Path = DATA_ROOT) -> pd.DataFrame:
    """카테고리별 학습 정상·평가 정상·평가 결함 장수와 결함 종류."""
    root = Path(root)
    rows = []
    for c in CATEGORIES:
        test_dirs = sorted(p for p in (root / c / "test").iterdir() if p.is_dir())
        defect_types = [p.name for p in test_dirs if p.name != "good"]
        rows.append({
            "category": c,
            "kind": "texture" if c in TEXTURES else "object",
            "train_good": len(list((root / c / "train" / "good").glob("*.png"))),
            "test_good": len(list((root / c / "test" / "good").glob("*.png"))),
            "test_defect": sum(len(list((root / c / "test" / d).glob("*.png"))) for d in defect_types),
            "n_defect_types": len(defect_types),
            "defect_types": ", ".join(defect_types),
        })
    return pd.DataFrame(rows).set_index("category")


# ---------------------------------------------------------------- PatchCore (Anomalib)
def make_lowmem_patchcore(**kwargs):
    """GPU 메모리가 작을 때(8GB 등) 쓰는 PatchCore.

    Anomalib 기본 구현은 학습 이미지의 패치 특징(screw 기준 약 2GB, float32)을 모은 뒤 한 번에 이어 붙여
    잠시 두 배의 메모리를 씁니다. 여기서는 패치 특징만 float16으로 보관하고, 코어셋 선택의 무작위 투영을
    나눠서 계산합니다. 신경망 계산과 최종 메모리 뱅크는 float32 그대로입니다.
    """
    from anomalib.models import Patchcore
    from anomalib.models.components.sampling import KCenterGreedy

    class _ChunkedKCenter(KCenterGreedy):
        def select_coreset_idxs(self) -> list[int]:
            self.model.fit(self.embedding)                  # 투영 행렬 생성 (크기만 사용)
            self.features = torch.cat([self.model.transform(self.embedding[i:i + 65536].float())
                                       for i in range(0, self.n_observations, 65536)])
            self.reset_distances()
            idx = torch.randint(high=self.n_observations, size=(1,), device=self.features.device).squeeze()
            selected = [idx]
            for _ in range(self.coreset_size - 1):
                self.update_distances(cluster_center=idx)
                self.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), -torch.inf)
                idx = self.get_new_idx()
                selected.append(idx)
            return torch.stack(selected).cpu().tolist()

    class LowMemPatchcore(Patchcore):
        def training_step(self, batch, *args, **kwargs):
            out = super().training_step(batch, *args, **kwargs)
            self.model.embedding_store[-1] = self.model.embedding_store[-1].half()
            return out

        def fit(self) -> None:
            store = self.model.embedding_store
            bank = torch.vstack(store)
            store.clear()
            sampler = _ChunkedKCenter(embedding=bank, sampling_ratio=self.coreset_sampling_ratio)
            self.model.memory_bank = sampler.sample_coreset().float()
            del bank, sampler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return LowMemPatchcore(**kwargs)


def make_evaluator():
    """PRO(AUPRO)만 Anomalib으로 계산한다.

    이미지·픽셀 AUROC는 scikit-learn으로 직접 계산한다. torchmetrics의 AUROC는 0~1 밖의 점수를
    로짓으로 보고 시그모이드를 씌우는데, 정규화하지 않은 PatchCore 점수(30~60 등)는 이때 모두 1에
    가까워져 AUROC가 0.5로 잘못 나왔다(실측).
    """
    from anomalib.metrics import AUPRO, Evaluator

    return Evaluator(test_metrics=[AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")])


def run_patchcore(category: str, coreset_ratio: float = 0.1, root: str | Path = DATA_ROOT,
                  keep_maps: bool = False, lowmem: bool = True) -> tuple[dict, pd.DataFrame, dict | None]:
    """PatchCore를 한 카테고리에 학습·평가한다.

    반환: (지표·시간 dict, 이미지별 점수 표, 이상 지도 dict 또는 None)
    점수 정규화(post_processor)는 끈다. Anomalib 기본 설정은 평가 데이터로 정규화·임계값을 정하기 때문이다.
    """
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning.pytorch import seed_everything

    seed_everything(SEED, verbose=False)
    dm = MVTecAD(root=str(root), category=category, train_batch_size=32, eval_batch_size=32, num_workers=0)
    make = make_lowmem_patchcore if lowmem else Patchcore
    model = make(backbone="wide_resnet50_2", layers=("layer2", "layer3"),
                 coreset_sampling_ratio=coreset_ratio, num_neighbors=9,
                 evaluator=make_evaluator(), post_processor=False, visualizer=False)
    engine = Engine(default_root_dir=str(WORK_DIR), logger=False, enable_progress_bar=False,
                    enable_model_summary=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    fit_sec = time.perf_counter() - t0
    metrics = engine.test(model=model, datamodule=dm, verbose=False)[0]

    from sklearn.metrics import roc_auc_score

    preds = engine.predict(model=model, datamodule=dm, return_predictions=True)
    rows, maps, all_maps, all_masks = [], {}, [], []
    for b in preds:
        amap = b.anomaly_map.squeeze(1).cpu().numpy().astype(np.float32)
        mask = b.gt_mask.cpu().numpy().astype(bool)
        all_maps.append(amap); all_masks.append(mask)
        for i, path in enumerate(b.image_path):
            p = Path(path)
            rel = p.relative_to(Path(root).resolve() / category).as_posix() if p.is_absolute() else p.as_posix()
            rows.append({"path": rel, "defect_type": p.parent.name,
                         "label": int(b.gt_label[i]), "score": float(b.pred_score[i])})
            if keep_maps:
                maps[rel] = amap[i].astype(np.float16)
    scores = pd.DataFrame(rows)
    all_maps, all_masks = np.concatenate(all_maps), np.concatenate(all_masks)

    infer_ms = _time_inference(model, dm)
    out = {
        "category": category, "coreset_ratio": coreset_ratio,
        "image_AUROC": round(float(roc_auc_score(scores["label"], scores["score"])), 4),
        "pixel_AUROC": round(float(roc_auc_score(all_masks.ravel(), all_maps.ravel())), 4),
        "pixel_AUPRO": round(float(metrics["pixel_AUPRO"]), 4),
        "fit_sec": round(fit_sec, 1),
        "memory_bank": int(model.model.memory_bank.shape[0]),
        "feature_dim": int(model.model.memory_bank.shape[1]),
        "infer_ms_per_image": infer_ms,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "n_test": len(scores),
    }
    del engine, model, preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, scores, (maps if keep_maps else None)


@torch.no_grad()
def _time_inference(model, dm, repeats: int = 5) -> float:
    """배치 32장 추론(특징 추출 + 메모리 뱅크 검색)의 1장당 시간(ms). 데이터 읽기 시간은 뺀다."""
    batch = next(iter(dm.test_dataloader()))                # 데이터 모듈이 이미 256×256으로 바꿔 준 이미지
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.model.to(device).eval()                     # Lightning은 학습이 끝나면 모델을 CPU로 옮겨 둔다
    x = model.pre_processor.transform(batch.image.to(device))
    net(x)                                         # 준비 실행
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / repeats / len(x) * 1000, 2)


# ---------------------------------------------------------------- 기준 모델: 이미지 전체 특징의 최근접 거리
def _load_images(paths, size: int = IMAGE_SIZE) -> torch.Tensor:
    from PIL import Image

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        out.append((t - mean) / std)
    return torch.stack(out)


@torch.no_grad()
def global_knn_scores(category: str, root: str | Path = DATA_ROOT, device: str | None = None) -> pd.DataFrame:
    """같은 사전학습 CNN(wide_resnet50_2)의 layer2·layer3 특징을 이미지 전체로 평균한 뒤,
    학습 정상 이미지 중 가장 가까운 것과의 거리를 이상 점수로 쓴다. 위치 정보(Heatmap)는 없다."""
    import torchvision

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(root)
    net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    net = net.to(device).eval()

    def embed(paths):
        feats = []
        for i in range(0, len(paths), 32):
            x = _load_images(paths[i:i + 32]).to(device)
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            l2 = net.layer2(net.layer1(x))
            l3 = net.layer3(l2)
            feats.append(torch.cat([l2.mean((2, 3)), l3.mean((2, 3))], 1).cpu())
        return torch.cat(feats)

    train = sorted((root / category / "train" / "good").glob("*.png"))
    test = sorted((root / category / "test").glob("*/*.png"))
    bank, q = embed(train), embed(test)
    dist = torch.cdist(q, bank).min(1).values.numpy()
    return pd.DataFrame({"path": [p.relative_to(root / category).as_posix() for p in test],
                         "defect_type": [p.parent.name for p in test],
                         "label": [int(p.parent.name != "good") for p in test],
                         "score": dist})


# ---------------------------------------------------------------- 임계값: 평가 데이터를 반으로 나눠 정하고 확인
def threshold_holdout(scores: pd.DataFrame, seed: int = SEED) -> dict:
    """평가 이미지를 정상·결함 비율을 유지하며 반으로 나눈다(A: 임계값 결정, B: 확인).

    정책 1: A에서 F1이 가장 큰 임계값. 정책 2: A의 결함을 하나도 놓치지 않는 가장 높은 임계값.
    """
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split

    a, b = train_test_split(scores, test_size=0.5, stratify=scores["label"], random_state=seed)
    p, r, th = precision_recall_curve(a["label"], a["score"])
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    thr_f1 = float(th[np.argmax(f1)])
    thr_zero_miss = float(a.loc[a["label"] == 1, "score"].min())

    def at(t):
        pred = b["score"] >= t
        tp = int((pred & (b["label"] == 1)).sum()); fp = int((pred & (b["label"] == 0)).sum())
        fn = int((~pred & (b["label"] == 1)).sum()); tn = int((~pred & (b["label"] == 0)).sum())
        return {"threshold": round(t, 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "recall": round(tp / max(tp + fn, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
                "fpr": round(fp / max(fp + tn, 1), 3)}

    return {"n_A": len(a), "n_B": len(b), "B_defect": int(b["label"].sum()), "B_good": int((b["label"] == 0).sum()),
            "F1최대": at(thr_f1), "미검출0": at(thr_zero_miss)}


def env_info() -> dict:
    import platform

    import sklearn

    info = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        import anomalib
        import torchvision

        info.update(anomalib=anomalib.__version__, torchvision=torchvision.__version__)
    except ImportError:
        pass
    return info
