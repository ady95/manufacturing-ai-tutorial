"""PHM 2016 CMP 가상계측 실습 공용 코드.

책 7~10장의 Notebook이 함께 사용한다.
- load_cmp          : 원본 zip을 내려받아(처음 한 번) 센서 시계열과 제거율을 읽는다
- add_group         : 단계·챔버 조합으로 공정 그룹(A123/A456/B)을 붙인다
- extract_features  : 웨이퍼·단계 단위 통계 특징
- make_models       : 비교할 회귀 모델
- regression_report : 전체·그룹별 MAE/RMSE/R²
- make_sequences    : 10장 딥러닝용 고정 길이 시계열 텐서

원본 다운로드 링크(phmsociety.org/sites/...)는 2026-11 확인 시 404였다.
그래서 인터넷 아카이브(Wayback Machine)에 보존된 같은 파일을 받는다.
"""
from __future__ import annotations

import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42

_WB = "https://web.archive.org/web/{ts}id_/https://www.phmsociety.org/sites/phmsociety.org/files/{name}"
CMP_FILES = {
    "cmp_train_test.zip": _WB.format(ts="20180413091420", name="2016%20PHM%20DATA%20CHALLENGE%20CMP%20DATA%20SET.zip"),
    "cmp_validation.zip": _WB.format(ts="20180413091303", name="2016%20PHM%20DATA%20CHALLENGE%20CMP%20VALIDATION%20DATA%20SET.zip"),
    "answers.zip": _WB.format(ts="20180413091339", name="PHM16TestValidationAnswers.zip"),
}
KEY = ["WAFER_ID", "STAGE"]

# 원본 25개 열 중 식별·시각 열을 뺀 센서 열
USAGE_COLS = [
    "USAGE_OF_BACKING_FILM", "USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_DRESSER_TABLE", "USAGE_OF_MEMBRANE", "USAGE_OF_PRESSURIZED_SHEET",
]
PROCESS_COLS = [
    "PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE",
    "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION", "DRESSING_WATER_STATUS",
]


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def _download(data_dir: Path):
    zdir = data_dir / "zip"
    zdir.mkdir(parents=True, exist_ok=True)
    for name, url in CMP_FILES.items():
        dst = zdir / name
        if not dst.exists():
            print("내려받는 중:", name)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                dst.write_bytes(resp.read())
        with zipfile.ZipFile(dst) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(data_dir / "raw", members)


def _read_split(folder: Path, split: str) -> pd.DataFrame:
    parts = []
    for f in sorted(folder.glob(f"CMP-{split}-*.csv")):
        x = pd.read_csv(f)
        if len(x):                       # 헤더만 있는 빈 파일이 섞여 있다
            parts.append(x)
    return pd.concat(parts, ignore_index=True)


def load_cmp(data_dir: str | Path = "data/phm2016"):
    """세 세트(training/validation/test)의 센서 시계열과 제거율을 읽는다.

    반환: {"train": (센서 DataFrame, 제거율 DataFrame), "val": (...), "test": (...)}
    validation·test 제거율은 대회 종료 후 공개된 정답 파일에서 읽는다.
    """
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    if not (raw / "PHM16TestValidationAnswers").exists():
        _download(data_dir)
    tdir = raw / "2016 PHM DATA CHALLENGE CMP DATA SET"
    vdir = raw / "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET"
    adir = raw / "PHM16TestValidationAnswers"
    return {
        "train": (_read_split(tdir / "CMP-data" / "training", "training"),
                  pd.read_csv(tdir / "CMP-training-removalrate.csv")),
        "val": (_read_split(vdir / "validation", "validation"),
                pd.read_csv(adir / "orig_CMP-validation-removalrate.csv")),
        "test": (_read_split(tdir / "CMP-data" / "test", "test"),
                 pd.read_csv(adir / "orig_CMP-test-removalrate.csv")),
    }


def add_group(df: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    """웨이퍼·단계마다 공정 그룹을 붙인다.

    A123: 단계 A, 챔버 1~3 / A456: 단계 A, 챔버 4~6 / B: 단계 B (챔버 4~6)
    """
    first_ch = sensors.groupby(KEY)["CHAMBER"].min().rename("first_chamber")
    out = df.merge(first_ch.reset_index(), on=KEY, how="left")
    out["grp"] = np.where(out["STAGE"] == "B", "B",
                          np.where(out["first_chamber"] <= 3, "A123", "A456"))
    return out.drop(columns="first_chamber")


# ---------------------------------------------------------------------------
# 특징
# ---------------------------------------------------------------------------
def extract_features(sensors: pd.DataFrame) -> pd.DataFrame:
    """웨이퍼·단계(WAFER_ID, STAGE)마다 한 행의 통계 특징을 만든다.

    - 메타: 시작 시각(t0), 소요 시간, 행 수
    - 소모품 사용량(USAGE_*): 시작 시점 값 (공정 중 거의 변하지 않는 누적 카운터)
    - 공정 변수: 평균·표준편차·최솟값·최댓값·중앙값, 0이 아닌 구간의 평균
    """
    s = sensors.sort_values(KEY + ["TIMESTAMP"])
    g = s.groupby(KEY, sort=False)
    meta = pd.DataFrame({
        "t0": g["TIMESTAMP"].min(),
        "duration": g["TIMESTAMP"].max() - g["TIMESTAMP"].min(),
        "n_rows": g.size(),
    })
    usage = g[USAGE_COLS].first().add_suffix("_start")
    stats = g[PROCESS_COLS].agg(["mean", "std", "min", "max", "median"])
    stats.columns = [f"{c}_{st}" for c, st in stats.columns]
    active = s[PROCESS_COLS].where(s[PROCESS_COLS] != 0)
    active[KEY] = s[KEY]
    act = active.groupby(KEY, sort=False)[PROCESS_COLS].mean().add_suffix("_active").fillna(0.0)
    feats = pd.concat([meta, usage, stats, act], axis=1).reset_index()
    feats["stage_B"] = (feats["STAGE"] == "B").astype(int)
    return feats


def build_table(sensors: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    """특징 + 제거율 + 공정 그룹을 한 테이블로 결합한다."""
    feats = extract_features(sensors)
    tab = feats.merge(rr, on=KEY, how="inner", validate="one_to_one")
    tab = add_group(tab, sensors)
    tab["grp_A123"] = (tab["grp"] == "A123").astype(int)
    return tab


def feature_columns(tab: pd.DataFrame, drop_time: bool = True) -> list[str]:
    """모델 입력 열. 기본값은 절대 시각(t0)을 제외한다 — 미래 시점에는 의미가 바뀌기 때문."""
    skip = set(KEY + ["AVG_REMOVAL_RATE", "grp"])
    if drop_time:
        skip.add("t0")
    return [c for c in tab.columns if c not in skip]


# ---------------------------------------------------------------------------
# 모델·평가
# ---------------------------------------------------------------------------
def make_models():
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    return {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "PLS": make_pipeline(StandardScaler(), PLSRegression(n_components=10)),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                              n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.7,
                                random_state=SEED, n_jobs=-1),
    }


def regression_report(y_true, y_pred, grp=None) -> pd.DataFrame:
    """전체와 공정 그룹별 MAE·RMSE·R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    def row(t, p):
        return {"n": len(t), "MAE": mean_absolute_error(t, p),
                "RMSE": float(np.sqrt(mean_squared_error(t, p))),
                "R2": r2_score(t, p) if len(t) > 1 else np.nan}

    rows = {"전체": row(y_true, y_pred)}
    if grp is not None:
        grp = np.asarray(grp)
        for gname in sorted(set(grp)):
            m = grp == gname
            rows[gname] = row(y_true[m], y_pred[m])
    return pd.DataFrame(rows).T


def time_knn_predict(train_tab: pd.DataFrame, target_tab: pd.DataFrame, k: int = 10) -> np.ndarray:
    """센서를 쓰지 않고 '같은 그룹에서 시각이 가장 가까운 학습 웨이퍼 k개'의 제거율로 예측한다.

    공식 분리처럼 학습·평가 웨이퍼가 같은 기간에 섞여 있을 때 정확도의 정체를 확인하는 비교 기준.
    """
    from sklearn.neighbors import KNeighborsRegressor

    out = np.zeros(len(target_tab))
    for gname in train_tab["grp"].unique():
        tr = train_tab[train_tab["grp"] == gname]
        mask = (target_tab["grp"] == gname).to_numpy()
        if mask.any():
            knn = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(tr[["t0"]], tr["AVG_REMOVAL_RATE"])
            out[mask] = knn.predict(target_tab.loc[mask, ["t0"]])
    return out


def simulate_policy(stream: pd.DataFrame, pred: np.ndarray, every: int, lam: float = 0.3,
                    init_last: dict | None = None, mode: str = "ewma"):
    """운영 시뮬레이션. stream 은 시각 순으로 정렬된 웨이퍼·단계 테이블.

    - 그룹마다 every 번째 웨이퍼마다 1장을 실제로 계측한다고 가정한다 (every=0: 계측 없음)
    - mode="ewma": 모델 예측 + 계측 잔차의 지수가중이동평균(EWMA) 편향 보정
    - mode="last": 모델 없이 같은 그룹의 직전 계측값을 그대로 사용
    반환: (보정된 예측, 계측 여부 배열). 평가는 계측하지 않은 웨이퍼로만 한다.
    """
    bias, count = {}, {}
    last = dict(init_last or {})
    out = np.zeros(len(stream))
    measured = np.zeros(len(stream), dtype=bool)
    for i, (g, yv, pv) in enumerate(zip(stream["grp"], stream["AVG_REMOVAL_RATE"], pred)):
        out[i] = last.get(g, pv) if mode == "last" else pv + bias.get(g, 0.0)
        c = count.get(g, 0)
        count[g] = c + 1
        if every and c % every == 0:
            measured[i] = True
            bias[g] = lam * (yv - pv) + (1 - lam) * bias.get(g, 0.0)
            last[g] = yv
    return out, measured


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t


# ---------------------------------------------------------------------------
# 10장: 시계열 텐서
# ---------------------------------------------------------------------------
def make_sequences(sensors: pd.DataFrame, keys: pd.DataFrame, length: int = 128,
                   cols: list[str] | None = None) -> np.ndarray:
    """웨이퍼·단계마다 시계열을 같은 길이(length)로 선형 보간해 (N, length, C) 배열을 만든다.

    keys 의 행 순서(WAFER_ID, STAGE)를 그대로 따른다.
    """
    cols = cols or (PROCESS_COLS + USAGE_COLS)
    s = sensors.sort_values(KEY + ["TIMESTAMP"])
    groups = {k: v[cols].to_numpy(float) for k, v in s.groupby(KEY, sort=False)}
    out = np.zeros((len(keys), length, len(cols)), dtype=np.float32)
    grid = np.linspace(0, 1, length)
    for i, k in enumerate(zip(keys["WAFER_ID"], keys["STAGE"])):
        arr = groups[k]
        src = np.linspace(0, 1, len(arr))
        for j in range(arr.shape[1]):
            out[i, :, j] = np.interp(grid, src, arr[:, j])
    return out
