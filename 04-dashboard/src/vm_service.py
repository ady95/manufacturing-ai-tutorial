"""15장 가상계측 모델 서비스 공용 코드: 모델 저장소(registry), 추론 서비스, 예측 기록.

PoC 2(PHM 2016 CMP)의 XGBoost 가상계측 모델을 서비스 형태로 감싼다.
- 7~9장 코드(../02-virtual-metrology/src/cmp_utils.py)의 특징 추출을 그대로 쓴다
- 모델과 함께 메타데이터(버전, 학습 기간, 특징 목록, 입력 범위, 예측 구간 폭, 라이브러리 버전)를 저장한다
- 예측·실측 기록은 SQLite 파일 하나에 쌓는다(모니터링 화면이 읽는다)
"""
from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
VM_DIR = BASE.parent / "02-virtual-metrology"


def _load_cmp_utils():
    """7~9장의 cmp_utils.py를 파일 경로로 불러온다(이 폴더에도 src 패키지가 있어 이름이 겹치기 때문)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("cmp_utils", VM_DIR / "src" / "cmp_utils.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


_cu = _load_cmp_utils()
KEY, PROCESS_COLS, SEED, USAGE_COLS = _cu.KEY, _cu.PROCESS_COLS, _cu.SEED, _cu.USAGE_COLS
add_group, build_table, extract_features = _cu.add_group, _cu.build_table, _cu.extract_features
feature_columns, load_cmp, make_models = _cu.feature_columns, _cu.load_cmp, _cu.make_models

REGISTRY = BASE / "models"
LOG_DB = BASE / "results" / "vm_log.sqlite"
LAM = 0.3                                    # EWMA 보정 가중치 (9장과 같은 값)
SENSOR_COLS = ["TIMESTAMP", "CHAMBER"] + USAGE_COLS + PROCESS_COLS
WATCH = ["USAGE_OF_DRESSER_TABLE_start", "USAGE_OF_DRESSER_start", "SLURRY_FLOW_LINE_C_mean", "duration"]


# ---------------------------------------------------------------- 데이터 (9장과 같은 구성)
def prepare_data(data_dir: Path = VM_DIR / "data" / "phm2016"):
    """세 세트를 합쳐 시각 순으로 정렬하고, 처음 40%를 초기 학습, 나머지를 운영 기간으로 나눈다."""
    data = load_cmp(data_dir)
    tabs = {s: build_table(*v) for s, v in data.items()}
    train = tabs["train"][tabs["train"]["AVG_REMOVAL_RATE"] < 1000]
    allt = pd.concat([train, tabs["val"], tabs["test"]], ignore_index=True).sort_values("t0").reset_index(drop=True)
    allt["day"] = (allt["t0"] - allt["t0"].min()) / 86400
    cut = int(len(allt) * 0.4)
    init, stream = allt.iloc[:cut].reset_index(drop=True), allt.iloc[cut:].reset_index(drop=True)
    sensors = pd.concat([v[0] for v in data.values()], ignore_index=True)
    return init, stream, sensors, feature_columns(train)


def sensor_payload(sensors_by_key, wafer_id, stage) -> list[dict]:
    """한 웨이퍼·단계의 원시 센서 기록을 API 입력 형식(행 목록)으로 만든다."""
    rows = sensors_by_key.get_group((wafer_id, stage))[SENSOR_COLS]
    return rows.to_dict("records")


def psi(ref, cur, bins: int = 10) -> float:
    """Population Stability Index (9장과 같은 정의)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    q = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((q - p) * np.log(q / p)))


# ---------------------------------------------------------------- 모델 저장소
def train_artifact(fit_tab: pd.DataFrame, cols: list[str], version: str, note: str = "") -> Path:
    """XGBoost를 학습해 models/<version>/ 에 모델과 메타데이터를 저장한다(아직 운영 반영 전)."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, cross_val_predict
    import sklearn
    import xgboost

    y = fit_tab["AVG_REMOVAL_RATE"].to_numpy()
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(fit_tab, groups=fit_tab["WAFER_ID"]))
    oof = cross_val_predict(make_models()["XGBoost"], fit_tab[cols], y, groups=fit_tab["WAFER_ID"], cv=folds)
    res = np.abs(y - oof)
    t0 = time.perf_counter()
    model = clone(make_models()["XGBoost"]).fit(fit_tab[cols], y)
    fit_sec = time.perf_counter() - t0
    out = REGISTRY / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {
        "version": version, "created": datetime.now().isoformat(timespec="seconds"), "note": note,
        "model": "XGBRegressor", "target": "AVG_REMOVAL_RATE", "features": cols,
        "train_n": int(len(fit_tab)), "train_t0_range": [float(fit_tab["t0"].min()), float(fit_tab["t0"].max())],
        "train_data_hash": hashlib.md5(pd.util.hash_pandas_object(fit_tab[KEY + ["t0"]], index=False).values.tobytes()).hexdigest(),
        "cv_mae": round(float(res.mean()), 4),
        "q90_by_group": {g: round(float(np.quantile(res[fit_tab["grp"] == g], 0.9)), 4) for g in fit_tab["grp"].unique()},
        "input_range": {c: [float(fit_tab[c].min()), float(fit_tab[c].max())] for c in cols if c.startswith("USAGE")},
        "psi_reference": {c: fit_tab[c].tolist() for c in WATCH},
        "fit_sec": round(fit_sec, 2),
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def registry_state() -> dict:
    f = REGISTRY / "registry.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"production": None, "history": []}


def promote(version: str, approver: str, reason: str) -> dict:
    """사람이 승인한 버전을 운영 모델로 지정한다. 누가·언제·왜 바꿨는지 이력을 남긴다."""
    state = registry_state()
    state["history"].append({"time": datetime.now().isoformat(timespec="seconds"), "from": state["production"],
                             "to": version, "approver": approver, "reason": reason})
    state["production"] = version
    (REGISTRY / "registry.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return state


def load_version(version: str):
    d = REGISTRY / version
    return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 추론 서비스
class VMService:
    """운영 모델로 예측하고, 실측값이 들어오면 그룹별 EWMA 보정값을 갱신한다. 모든 요청을 기록한다."""

    def __init__(self, log_db: Path = LOG_DB):
        self.log_db = Path(log_db)
        self.log_db.parent.mkdir(parents=True, exist_ok=True)
        self.bias: dict[str, float] = {}
        self._init_db()
        self.reload()

    def reload(self) -> str:
        """운영 모델을 다시 읽는다. 모델이 바뀌면 EWMA 보정값을 초기화한다(이전 모델의 잔차이므로)."""
        new = registry_state()["production"]
        if new != getattr(self, "version", None):
            self.bias = {}
        self.version = new
        self.model, self.meta = load_version(self.version)
        return self.version

    def _init_db(self):
        with sqlite3.connect(self.log_db) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS predictions (
                wafer_id TEXT, stage TEXT, grp TEXT, t0 REAL, model_version TEXT, pred_raw REAL, bias REAL,
                pred REAL, lo REAL, hi REAL, out_of_range TEXT, latency_ms REAL, logged_at TEXT)""")
            con.execute("""CREATE TABLE IF NOT EXISTS feedback (
                wafer_id TEXT, stage TEXT, grp TEXT, actual REAL, pred_raw REAL, pred REAL,
                model_version TEXT, bias_after REAL, logged_at TEXT)""")

    def predict(self, wafer_id: str, stage: str, rows: list[dict]) -> dict:
        t_start = time.perf_counter()
        sensors = pd.DataFrame(rows)
        sensors["WAFER_ID"], sensors["STAGE"] = wafer_id, stage
        feats = add_group(extract_features(sensors), sensors)
        feats["grp_A123"] = (feats["grp"] == "A123").astype(int)
        f = feats.iloc[0]
        grp = str(f["grp"])
        raw = float(self.model.predict(feats[self.meta["features"]])[0])
        b = self.bias.get(grp, 0.0)
        pred = raw + b
        q = self.meta["q90_by_group"].get(grp, max(self.meta["q90_by_group"].values()))
        oor = [c for c, (lo, hi) in self.meta["input_range"].items() if not (lo <= float(f[c]) <= hi)]
        out = {"wafer_id": wafer_id, "stage": stage, "group": grp, "model_version": self.version,
               "pred_raw": round(raw, 4), "bias": round(b, 4), "pred": round(pred, 4),
               "interval_90": [round(pred - q, 4), round(pred + q, 4)], "out_of_range": oor,
               "latency_ms": round((time.perf_counter() - t_start) * 1000, 2)}
        with sqlite3.connect(self.log_db) as con:
            con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (wafer_id, stage, grp, float(f["t0"]), self.version, raw, b, pred, pred - q, pred + q,
                         ",".join(oor), out["latency_ms"], datetime.now().isoformat(timespec="seconds")))
        return out

    def feedback(self, wafer_id: str, stage: str, actual: float) -> dict:
        """실측값으로 해당 그룹의 EWMA 보정값을 갱신한다. 가장 최근 예측과 짝을 맞춘다."""
        with sqlite3.connect(self.log_db) as con:
            row = con.execute("SELECT grp, pred_raw, pred FROM predictions WHERE wafer_id=? AND stage=? "
                              "ORDER BY rowid DESC LIMIT 1", (wafer_id, stage)).fetchone()
            if row is None:
                raise KeyError(f"예측 기록이 없습니다: {wafer_id}/{stage}")
            grp, raw, pred = row
            self.bias[grp] = LAM * (actual - raw) + (1 - LAM) * self.bias.get(grp, 0.0)
            con.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?)",
                        (wafer_id, stage, grp, actual, raw, pred, self.version, self.bias[grp],
                         datetime.now().isoformat(timespec="seconds")))
        return {"wafer_id": wafer_id, "stage": stage, "group": grp, "residual_raw": round(actual - raw, 4),
                "bias_after": round(self.bias[grp], 4)}
