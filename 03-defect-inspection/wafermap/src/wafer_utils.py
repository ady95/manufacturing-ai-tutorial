"""PoC 3-A. WM-811K 웨이퍼맵 결함 패턴 분류 공용 코드 (책 11~12장)."""
from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SEED = 42
SIZE = 64
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
KAGGLE_SLUG = "qingyi/wm811k-wafer-map"


# ---------------------------------------------------------------- 데이터 취득·로딩
def download_wm811k(data_dir: str | Path = "data/wm811k") -> Path:
    """Kaggle에서 LSWMD.pkl을 받는다. KAGGLE_API_TOKEN(또는 kaggle.json)이 필요하다."""
    data_dir = Path(data_dir)
    pkl = data_dir / "LSWMD.pkl"
    if pkl.exists():
        return pkl
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(data_dir), "--unzip"],
                   check=True)
    return pkl


class _LegacyUnpickler(pickle.Unpickler):
    """옛 pandas로 저장된 pickle을 최신 pandas에서 읽는다 (pandas.indexes → pandas.core.indexes)."""

    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes")
        return super().find_class(module, name)


def _first(x):
    """[['Training']] 처럼 감싼 라벨에서 문자열만 꺼낸다. 라벨이 없으면 None."""
    try:
        return str(x[0][0]) if len(x) and len(x[0]) else None
    except (TypeError, IndexError):
        return None


def load_raw(data_dir: str | Path = "data/wm811k") -> pd.DataFrame:
    pkl = download_wm811k(data_dir)
    with open(pkl, "rb") as f:
        df = _LegacyUnpickler(f, encoding="latin1").load()
    df["label"] = df["failureType"].map(_first)
    df["split"] = df["trianTestLabel"].map(_first)          # 원본 열 이름의 오타(trian)를 그대로 쓴다
    return df


def resize_map(m: np.ndarray, size: int = SIZE) -> np.ndarray:
    """0(바깥)·1(정상)·2(불량) 값을 보존하도록 최근접 보간으로 크기를 맞춘다."""
    return cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def prepare(data_dir: str | Path = "data/wm811k", size: int = SIZE, force: bool = False) -> dict:
    """라벨 있는 웨이퍼맵만 size×size로 줄여 npz로 캐시한다. 원본 2GB를 매번 읽지 않기 위함."""
    data_dir = Path(data_dir)
    cache = data_dir / f"wm{size}_labeled.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    df = load_raw(data_dir)
    df = df[df["label"].notna()].reset_index(drop=True)
    maps = np.stack([resize_map(m, size) for m in df["waferMap"]])
    out = {
        "maps": maps,
        "y": df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(np.int64),
        "split": df["split"].to_numpy().astype("U8"),
        "lot": df["lotName"].to_numpy().astype("U12"),
        "lot_num": df["lotName"].str.extract(r"(\d+)")[0].astype(int).to_numpy(),
        "H": df["waferMap"].map(lambda m: m.shape[0]).to_numpy(),
        "W": df["waferMap"].map(lambda m: m.shape[1]).to_numpy(),
        "n_defect": df["waferMap"].map(lambda m: int((m == 2).sum())).to_numpy(),
        "md5": df["waferMap"].map(
            lambda m: hashlib.md5(m.astype(np.uint8).tobytes() + str(m.shape).encode()).hexdigest()
        ).to_numpy().astype("U32"),
    }
    np.savez_compressed(cache, **out)
    return out


def dev_val_split(lot_num: np.ndarray, is_train: np.ndarray, val_frac: float = 0.2):
    """Training 안에서 Lot 번호가 큰 쪽 웨이퍼 val_frac을 검증으로 뗀다 (공식 Test가 나중 Lot이므로 같은 방향).

    한 Lot은 통째로 한쪽에만 들어간다. Lot 개수가 아니라 웨이퍼 수 기준으로 자른다
    (뒤쪽 Lot에 라벨 있는 웨이퍼가 몰려 있어, Lot 개수로 자르면 검증이 절반 가까이 된다).
    """
    cut = np.quantile(lot_num[is_train], 1 - val_frac, method="higher")
    tr = is_train & (lot_num < cut)
    va = is_train & (lot_num >= cut)
    return tr, va


# ---------------------------------------------------------------- OpenCV 특징 (11-2)
def _ring_sector_index(size: int = SIZE):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c) / (size / 2)
    ang = (np.arctan2(yy - c, xx - c) + np.pi) / (2 * np.pi)
    ring = np.minimum((r * 5).astype(int), 4)                    # 반지름 5등분
    sector = np.minimum((ang * 8).astype(int), 7)                # 방향 8등분
    return ring, sector


_RING, _SECTOR = _ring_sector_index()
FEATURE_NAMES = (["defect_ratio"] + [f"ring{i}" for i in range(5)] + [f"sector{i}" for i in range(8)]
                 + ["n_comp", "largest_area", "second_area", "largest_r", "largest_elong",
                    "largest_extent", "clean_ratio", "edge_vs_center"])


def handcrafted_features(m: np.ndarray) -> np.ndarray:
    """웨이퍼맵 한 장에서 규칙 기반 특징 22개를 만든다."""
    wafer = m > 0
    defect = (m == 2).astype(np.uint8)
    n_wafer = max(int(wafer.sum()), 1)
    feats = [defect.sum() / n_wafer]
    for i in range(5):
        zone = wafer & (_RING == i)
        feats.append(defect[zone].mean() if zone.any() else 0.0)
    outer = wafer & (_RING >= 3)
    for i in range(8):
        zone = outer & (_SECTOR == i)
        feats.append(defect[zone].mean() if zone.any() else 0.0)
    # 잡음 제거: 중앙값 필터 → 열림 연산. 흩어진 점은 지우고 덩어리만 남긴다
    clean = cv2.medianBlur(defect * 255, 3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(clean.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
    order = np.argsort(-areas) if len(areas) else []
    largest = areas[order[0]] / n_wafer if len(areas) else 0.0
    second = areas[order[1]] / n_wafer if len(areas) > 1 else 0.0
    r_norm = elong = extent = 0.0
    if len(areas):
        k = order[0] + 1
        cy, cx = cent[k][1], cent[k][0]
        c = (m.shape[0] - 1) / 2
        r_norm = np.hypot(cy - c, cx - c) / (m.shape[0] / 2)
        ys, xs = np.nonzero(lab == k)
        if len(xs) >= 3:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])))
            elong = float(np.sqrt(ev[-1] / max(ev[0], 1e-6)))
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        extent = areas[order[0]] / max(w * h, 1)
    edge = defect[wafer & (_RING == 4)].mean() if (wafer & (_RING == 4)).any() else 0.0
    center = defect[wafer & (_RING == 0)].mean() if (wafer & (_RING == 0)).any() else 0.0
    feats += [len(areas), largest, second, r_norm, elong, extent, clean.sum() / n_wafer, edge - center]
    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------- 평가
def classification_report_dict(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)),
                                         zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "f1": round(float(f[i]), 3), "support": int(s[i])} for i, c in enumerate(CLASSES)},
    }


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def env_info() -> dict:
    import platform

    import sklearn

    info = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "opencv": cv2.__version__}
    try:
        import torch
        import torchvision

        info.update(torch=torch.__version__, torchvision=torchvision.__version__,
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except ImportError:
        pass
    return info
