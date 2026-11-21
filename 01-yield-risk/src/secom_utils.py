"""SECOM 불량 위험 예측 실습 공용 코드.

책 4~6장의 Notebook이 함께 사용한다.
- load_secom      : UCI 원본을 내려받아 특성(X), 라벨(y), 시각(ts)을 반환
- dev_test_split  : 시간 순서대로 개발용/최종 평가용 분리
- time_folds, oof_predict : 개발 데이터 안에서의 시간 교차검증
- make_preprocessor : 학습 데이터에만 적합되는 전처리 파이프라인
- make_models     : 비교할 모델 4종
- evaluate        : 불균형 분류 지표 계산
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SEED = 42


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def load_secom(data_dir: str | Path = "data/secom"):
    """SECOM을 내려받아(처음 한 번) 읽는다.

    반환: X (1567×590 DataFrame), y (1=불량, 0=양품), ts (측정 시각)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / "secom.data").exists():
        with urllib.request.urlopen(SECOM_URL) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(data_dir)

    X = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"f{i:03d}" for i in range(X.shape[1])]
    labels = pd.read_csv(
        data_dir / "secom_labels.data", sep=r"\s+", header=None,
        names=["label", "timestamp"],
    )
    y = (labels["label"] == 1).astype(int).rename("fail")  # 원본: -1=Pass, 1=Fail
    ts = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S").rename("ts")
    return X, y, ts


def dev_test_split(X, y, ts, test_size=0.2):
    """시각 순으로 정렬한 뒤 앞부분은 개발용(dev), 마지막 test_size는 최종 평가용(test)으로 나눈다.

    평가 데이터는 모델·임계값을 모두 정한 뒤 마지막에 한 번만 사용한다.
    """
    order = np.argsort(ts.values, kind="stable")
    cut = int(len(order) * (1 - test_size))
    dev, test = order[:cut], order[cut:]
    return (
        (X.iloc[dev].reset_index(drop=True), y.iloc[dev].reset_index(drop=True), ts.iloc[dev].reset_index(drop=True)),
        (X.iloc[test].reset_index(drop=True), y.iloc[test].reset_index(drop=True), ts.iloc[test].reset_index(drop=True)),
    )


def time_folds(n: int, start: float = 0.4, k: int = 3):
    """확장 윈도(expanding window) 시간 교차검증 분할.

    처음 start 비율로 학습을 시작하고, 이후 구간을 k개 블록으로 나눠
    '과거 전체로 학습 → 바로 다음 블록 예측'을 반복한다.
    """
    bounds = np.linspace(int(n * start), n, k + 1).astype(int)
    return [(np.arange(0, bounds[i]), np.arange(bounds[i], bounds[i + 1])) for i in range(k)]


def oof_predict(make_model, X, y, folds):
    """교차검증 각 분할의 평가 블록 예측을 이어 붙인다 (out-of-fold 예측).

    make_model(pos_weight) 는 새 모델을 반환하는 함수다.
    반환: (예측 대상 인덱스, 실제 라벨, 예측 확률, 분할별 PR-AUC 목록)
    """
    idx_all, y_all, p_all, fold_ap = [], [], [], []
    for tr, te in folds:
        y_tr = y.iloc[tr]
        model = make_model((y_tr == 0).sum() / y_tr.sum())
        model.fit(X.iloc[tr], y_tr)
        p = model.predict_proba(X.iloc[te])[:, 1]
        idx_all.append(te)
        y_all.append(y.iloc[te].to_numpy())
        p_all.append(p)
        fold_ap.append(average_precision_score(y.iloc[te], p))
    return np.concatenate(idx_all), np.concatenate(y_all), np.concatenate(p_all), fold_ap


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
class DropHighMissing(BaseEstimator, TransformerMixin):
    """학습 데이터에서 결측 비율이 threshold를 넘는 열을 제거한다."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        miss = np.isnan(np.asarray(X, dtype=float)).mean(axis=0)
        self.keep_ = np.where(miss <= self.threshold)[0]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else getattr(self, "feature_names_in_", None)
        if names is None:
            names = np.array([f"x{i}" for i in range(int(self.keep_.max()) + 1)])
        return np.asarray(names)[self.keep_]


def make_preprocessor(scale: bool = True, missing_threshold: float = 0.5) -> Pipeline:
    """결측 많은 열 제거 → 중앙값 대치 → 상수 열 제거 → (선택) 표준화."""
    steps = [
        ("drop_missing", DropHighMissing(missing_threshold)),
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold(0.0)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def make_models(pos_weight: float, seed: int = SEED) -> dict:
    """비교할 모델 4종. pos_weight = 양품 수 / 불량 수 (학습 데이터 기준)."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000)),
        ]),
        "RandomForest": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed)),
        ]),
        "XGBoost": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.5, scale_pos_weight=pos_weight, eval_metric="aucpr",
                n_jobs=-1, random_state=seed)),
        ]),
        "LightGBM": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                scale_pos_weight=pos_weight, verbose=-1, n_jobs=-1, random_state=seed)),
        ]),
    }


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """불균형 분류 지표. 임계값 기반 지표는 threshold로 이진화해 계산한다."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": (tp + tn) / len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci(y_true, proba, metric=average_precision_score, n=1000, seed=SEED):
    """평가 데이터 재표집으로 지표의 95% 신뢰구간을 구한다."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        vals.append(metric(y_true[idx], proba[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def timed_fit_predict(model, X_train, y_train, X_eval):
    """학습 시간과 예측 시간을 함께 잰다."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = model.predict_proba(X_eval)[:, 1]
    pred_sec = time.perf_counter() - t0
    return proba, fit_sec, pred_sec
