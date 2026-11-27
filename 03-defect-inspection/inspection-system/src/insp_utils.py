"""14장 지능형 판독 시스템 공용 코드: YOLO 결함 검출, PatchCore와의 비교, 3단계 판정.

데이터는 13장과 같은 MVTec AD(CC BY-NC-SA 4.0, 비상업적 이용)를 쓰고, 13장 Notebook이 만든
PatchCore 점수·이상 지도(../anomaly-detection/results/13_runs)를 불러옵니다.
YOLO(Ultralytics)는 AGPL-3.0 라이선스입니다. 학습·연구 목적 외에 쓰려면 라이선스 조건을 확인하세요.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # 실행 중에 패키지를 멋대로 설치하지 않게 함

import cv2
import numpy as np
import pandas as pd

SEED = 42
FOCUS = ["bottle", "screw", "carpet"]
MVTEC = Path("../anomaly-detection/data/mvtec_ad")
RUNS13_OWN = Path("../anomaly-detection/results/13_runs")         # 13장 Notebook을 직접 돌린 결과
RUNS13_BOOK = Path("../anomaly-detection/results/13_runs_book")   # 책 실측 결과 (예제 저장소에 포함)


def _pick_runs13() -> Path:
    """13장 결과를 직접 만들었으면(15개 카테고리 점수 + 이상 지도 3개) 그것을, 아니면 책 실측 결과를 쓴다."""
    own_ok = (len(list(RUNS13_OWN.glob("*_pc10.csv"))) >= 15
              and all((RUNS13_OWN / f"{c}_maps.npz").exists() for c in FOCUS))
    return RUNS13_OWN if own_ok else RUNS13_BOOK


RUNS13 = _pick_runs13()
WORK = Path("work")                                  # YOLO 데이터·체크포인트 (저장소에 올리지 않음)
YOLO_CFG = dict(epochs=100, imgsz=640, batch=16, workers=0, amp=False, seed=SEED, deterministic=True,
                plots=False, verbose=False, val=False)    # 고정 100 epoch, 마지막 가중치 사용
CONF = 0.25                                          # YOLO 기본 신뢰도 기준 (결과를 보기 전에 정함)


# ---------------------------------------------------------------- 13장 결과 불러오기와 A/B 분할
def load_pc_scores(category: str, ratio: str = "pc10") -> pd.DataFrame:
    return pd.read_csv(RUNS13 / f"{category}_{ratio}.csv")


def holdout_split(scores: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """13-4와 똑같은 A/B 분할(정상·불량 비율 유지, 반반)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(scores, test_size=0.5, stratify=scores["label"], random_state=seed)


def f1_threshold(a: pd.DataFrame) -> float:
    """A에서 F1이 가장 큰 임계값 (13-4의 'F1 최대' 정책)."""
    from sklearn.metrics import precision_recall_curve

    p, r, th = precision_recall_curve(a["label"], a["score"])
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(th[np.argmax(f1)])


# ---------------------------------------------------------------- 마스크 → YOLO 박스 라벨
def mask_path(category: str, rel: str) -> Path:
    p = Path(rel)
    return MVTEC / category / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"


def mask_to_boxes(mask_file: Path, min_area: int = 16) -> list[tuple[float, float, float, float]]:
    """결함 마스크의 연결 요소마다 외곽 상자를 만든다. YOLO 형식(중심 x, 중심 y, 너비, 높이, 0~1)."""
    m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    h, w = m.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < min_area:
            continue
        boxes.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return boxes


def build_yolo_dataset(category: str, a: pd.DataFrame, b: pd.DataFrame, name: str,
                       exclude_types: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """A로 학습용, B로 평가용 YOLO 데이터 폴더를 만든다. 정상 이미지는 라벨 파일이 빈 배경 이미지가 된다.

    반환: (학습용 data.yaml — 검증도 A, 평가용 data.yaml — 검증이 B)
    """
    root = WORK / "datasets" / name
    if root.exists():
        root = _safe_rmtree(root)
    for split, df in [("train", a[~a["defect_type"].isin(exclude_types)]), ("test", b)]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for _, r in df.iterrows():
            stem = r["path"].replace("/", "__").removesuffix(".png")
            shutil.copy(MVTEC / category / r["path"], root / "images" / split / f"{stem}.png")
            lines = [] if r["label"] == 0 else [
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in mask_to_boxes(mask_path(category, r["path"]))]
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    common = f"path: {root.resolve().as_posix()}\nnames:\n  0: defect\n"
    y_train, y_eval = root / "train.yaml", root / "eval.yaml"
    y_train.write_text(common + "train: images/train\nval: images/train\n", encoding="utf-8")
    y_eval.write_text(common + "train: images/train\nval: images/test\n", encoding="utf-8")
    return y_train, y_eval


def _safe_rmtree(root: Path) -> Path:
    """Windows에서는 백신·검색 색인이 방금 만든 파일을 잠깐 잡고 있어 삭제가 실패할 수 있다(실측).
    몇 번 다시 시도하고, 그래도 안 되면 새 이름의 폴더를 쓴다."""
    for _ in range(5):
        try:
            shutil.rmtree(root)
            return root
        except PermissionError:
            time.sleep(3)
    return root.with_name(f"{root.name}_{int(time.time())}")


# ---------------------------------------------------------------- YOLO 학습·평가
def train_yolo(data_yaml: Path, name: str, weights: str = "yolo26n.pt", retries: int = 1):
    """같은 이유로 체크포인트 저장이 실패하면(I/O operation on closed file) 한 번 더 학습한다."""
    from ultralytics import YOLO

    for attempt in range(retries + 1):
        model = YOLO(weights)
        t0 = time.perf_counter()
        try:
            model.train(data=str(data_yaml), project=str((WORK / "yolo").resolve()), name=name, exist_ok=True,
                        **YOLO_CFG)
            break
        except (ValueError, OSError, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(10)
    sec = time.perf_counter() - t0
    last = WORK / "yolo" / name / "weights" / "last.pt"
    return YOLO(str(last)), round(sec, 1)


def yolo_image_scores(model, category: str, df: pd.DataFrame) -> pd.DataFrame:
    """이미지마다 가장 높은 상자 신뢰도를 이미지 점수로 쓴다(상자가 없으면 0). 상자도 함께 저장."""
    rows = []
    paths = [str(MVTEC / category / p) for p in df["path"]]
    for i in range(0, len(paths), 16):
        for res, (_, r) in zip(model.predict(paths[i:i + 16], imgsz=YOLO_CFG["imgsz"], conf=0.001, verbose=False),
                               df.iloc[i:i + 16].iterrows()):
            conf = res.boxes.conf.cpu().numpy()
            xyxyn = res.boxes.xyxyn.cpu().numpy()
            keep = conf >= CONF
            rows.append({**r[["path", "defect_type", "label"]].to_dict(),
                         "score": float(conf.max()) if len(conf) else 0.0,
                         "boxes": json.dumps(np.round(np.c_[xyxyn[keep], conf[keep]], 4).tolist())})
    return pd.DataFrame(rows)


def yolo_box_metrics(model, eval_yaml: Path) -> dict:
    m = model.val(data=str(eval_yaml), split="val", imgsz=YOLO_CFG["imgsz"], batch=16, workers=0, plots=False,
                  verbose=False)
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "box_precision": round(float(m.box.mp), 4), "box_recall": round(float(m.box.mr), 4)}


def yolo_speed(model, category: str, df: pd.DataFrame, repeats: int = 3) -> dict:
    """배치 16장 예측의 1장당 시간(ms). Ultralytics가 재는 전처리·추론·후처리 시간을 평균한다."""
    paths = [str(MVTEC / category / p) for p in df["path"].iloc[:16]]
    model.predict(paths, imgsz=YOLO_CFG["imgsz"], verbose=False)              # 준비 실행
    sp = []
    for _ in range(repeats):
        for r in model.predict(paths, imgsz=YOLO_CFG["imgsz"], verbose=False):
            sp.append(r.speed)
    return {k: round(float(np.mean([s[k] for s in sp])), 2) for k in ["preprocess", "inference", "postprocess"]}


# ---------------------------------------------------------------- 학습 작업 (결과는 results/14_runs에 저장)
RUNS14 = Path("results/14_runs")


def job_name(category: str, exclude: str | None = None) -> str:
    return f"{category}_full" if exclude is None else f"{category}_wo_{exclude}"


def run_job(category: str, exclude: str | None = None) -> tuple[dict, pd.DataFrame]:
    """YOLO 한 번 학습·평가. 끝난 작업은 다시 하지 않는다."""
    name = job_name(category, exclude)
    f_json, f_csv = RUNS14 / f"{name}.json", RUNS14 / f"{name}.csv"
    if f_json.exists() and f_csv.exists():
        return json.loads(f_json.read_text(encoding="utf-8")), pd.read_csv(f_csv)
    RUNS14.mkdir(parents=True, exist_ok=True)
    s = load_pc_scores(category)
    a, b = holdout_split(s)
    yt, ye = build_yolo_dataset(category, a, b, name, exclude_types=() if exclude is None else (exclude,))
    model, sec = train_yolo(yt, name)
    scores = yolo_image_scores(model, category, b)
    out = {"name": name, "category": category, "exclude": exclude, "train_sec": sec,
           "n_train": int(len(a) - (a.defect_type == exclude).sum() if exclude else len(a)),
           "n_train_defect": int(((a.label == 1) & (a.defect_type != exclude)).sum())}
    if exclude is None:
        out.update(yolo_box_metrics(model, ye))
        out["speed_ms"] = yolo_speed(model, category, b)
    f_json.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    scores.to_csv(f_csv, index=False)
    return out, scores


def all_jobs() -> list[tuple[str, str | None]]:
    jobs = [(c, None) for c in FOCUS]
    for c in FOCUS:
        s = load_pc_scores(c)
        jobs += [(c, t) for t in sorted(s.loc[s.label == 1, "defect_type"].unique())]
    return jobs


def run_all_jobs_in_subprocesses() -> None:
    """학습마다 새 파이썬 프로세스를 띄운다. 한 커널에서 여러 번 학습하면 PC 메모리가 쌓여
    MemoryError가 났기 때문이다(실측, 메모리 32GB PC)."""
    import subprocess
    import sys

    for c, ex in all_jobs():
        if (RUNS14 / f"{job_name(c, ex)}.json").exists():
            continue
        code = f"import src.insp_utils as U; U.run_job({c!r}, {ex!r})"
        for attempt in range(2):                     # 파일 잠금 등 일시적 실패에 대비해 한 번 더 시도
            if subprocess.run([sys.executable, "-c", code]).returncode == 0:
                break
        else:
            raise RuntimeError(f"학습 실패: {job_name(c, ex)}")


# ---------------------------------------------------------------- 위치 적중
def box_hits_mask(boxes_json: str, mask_file: Path) -> bool:
    """신뢰도 기준을 넘은 상자 중 하나라도 정답 결함 픽셀을 포함하면 적중."""
    m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0
    h, w = m.shape
    for x1, y1, x2, y2, _ in json.loads(boxes_json):
        if m[int(y1 * h):max(int(y2 * h), int(y1 * h) + 1), int(x1 * w):max(int(x2 * w), int(x1 * w) + 1)].any():
            return True
    return False


def heatmap_peak_hits_mask(amap: np.ndarray, mask_file: Path) -> bool:
    """PatchCore 이상 지도의 최고점이 정답 결함 픽셀 위에 있으면 적중."""
    m = cv2.resize(cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE), amap.shape[::-1],
                   interpolation=cv2.INTER_NEAREST) > 0
    y, x = np.unravel_index(np.argmax(amap), amap.shape)
    return bool(m[y, x])


def load_pc_maps(category: str) -> dict[str, np.ndarray]:
    z = np.load(RUNS13 / f"{category}_maps.npz")
    return {k.replace("__", "/"): z[k].astype(np.float32) for k in z.files}


# ---------------------------------------------------------------- 3단계 판정
def three_way(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """A에서 두 임계값을 정하고 B에 적용한다.

    t_low : A의 불량을 하나도 놓치지 않는 가장 높은 값(이보다 낮으면 자동 통과)
    t_high: A의 정상을 하나도 불량으로 보내지 않는 가장 낮은 값(이보다 높으면 자동 불량)
    둘 사이는 재검토. t_low가 t_high보다 크면(A에서 완전히 분리) 하나의 기준으로 합친다.
    """
    t_low = float(a.loc[a["label"] == 1, "score"].min())
    t_high = float(a.loc[a["label"] == 0, "score"].max()) + 1e-9
    if t_low > t_high:
        t_low = t_high = (t_low + t_high) / 2
    auto_pass = b["score"] < t_low
    auto_ng = b["score"] >= t_high
    review = ~auto_pass & ~auto_ng
    return {"t_low": round(t_low, 4), "t_high": round(t_high, 4), "n": len(b),
            "auto_pass": int(auto_pass.sum()), "review": int(review.sum()), "auto_ng": int(auto_ng.sum()),
            "missed_in_pass": int((auto_pass & (b["label"] == 1)).sum()),
            "good_in_ng": int((auto_ng & (b["label"] == 0)).sum()),
            "review_defect": int((review & (b["label"] == 1)).sum()),
            "review_good": int((review & (b["label"] == 0)).sum())}


def env_info() -> dict:
    import platform

    import sklearn
    import torch

    info = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "opencv": cv2.__version__, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return info
