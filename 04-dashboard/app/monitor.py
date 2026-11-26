"""15-3 실습: 가상계측 모니터링 화면 (Streamlit).

실행 (04-dashboard 폴더에서):
    streamlit run app/monitor.py

추론 API가 쌓은 예측·실측 기록(results/vm_log.sqlite)과 모델 저장소 이력(models/registry.json),
15장 Notebook의 주간 점검 결과(results/15_service.json)를 읽어 보여 줍니다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
LOG_DB = BASE / "results" / "vm_log.sqlite"
REGISTRY = BASE / "models" / "registry.json"
RESULT = BASE / "results" / "15_service.json"
TOL = 5.0                                     # 허용오차 ±5 (9장과 같은 가정값)

st.set_page_config(page_title="CMP 가상계측 모니터링", layout="wide")


@st.cache_data(ttl=10)
def load_log():
    with sqlite3.connect(LOG_DB) as con:
        pred = pd.read_sql("SELECT rowid AS seq, * FROM predictions", con)
        fb = pd.read_sql("SELECT * FROM feedback", con)
    pred["day"] = (pred["t0"] - pred["t0"].min()) / 86400
    fb = fb.merge(pred[["wafer_id", "stage", "day"]].drop_duplicates(["wafer_id", "stage"], keep="last"),
                  on=["wafer_id", "stage"], how="left")
    fb["err"] = (fb["actual"] - fb["pred"]).abs()
    fb["err_raw"] = (fb["actual"] - fb["pred_raw"]).abs()
    return pred, fb


if not LOG_DB.exists():
    st.warning("예측 기록이 없습니다. 15장 Notebook을 먼저 실행하세요.")
    st.stop()

pred, fb = load_log()
registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
events = json.loads(RESULT.read_text(encoding="utf-8"))["events"] if RESULT.exists() else []

st.title("CMP 가상계측 모니터링")
groups = st.sidebar.multiselect("공정 그룹", sorted(pred["grp"].unique()), default=sorted(pred["grp"].unique()))
window = st.sidebar.slider("이동 평균 창(실측 건수)", 10, 100, 30, step=10)
p, f = pred[pred["grp"].isin(groups)], fb[fb["grp"].isin(groups)]

# ---------------------------------------------------------------- 요약 지표
recent = f.tail(window)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("운영 모델", registry["production"].replace("cmp-xgb-", ""))
c2.metric("예측 건수", f"{len(p):,}")
c3.metric("실측 건수", f"{len(f):,}")
c4.metric(f"최근 {window}건 MAE(보정 후)", f"{recent['err'].mean():.2f}")
c5.metric("입력 범위 밖 비율", f"{(p['out_of_range'].fillna('').str.len() > 0).mean():.1%}")

# ---------------------------------------------------------------- 가상계측값과 실측값 추이
fig, ax = plt.subplots(figsize=(11, 3.6))
for g, d in p.groupby("grp"):
    ax.plot(d["day"], d["pred"], lw=0.6, alpha=0.6, label=f"VM {g}")
ax.scatter(f["day"], f["actual"], s=8, color="k", label="measured")
for h in registry["history"][1:]:
    first = p.loc[p["model_version"] == h["to"], "day"]
    if len(first):
        ax.axvline(first.min(), color="tab:green", ls=":", lw=1)
ax.set_xlabel("day"); ax.set_ylabel("removal rate"); ax.legend(fontsize=7, ncol=4)
st.subheader("가상계측값과 실측값")
st.pyplot(fig)

# ---------------------------------------------------------------- 실측 웨이퍼의 오차와 알람
f = f.assign(roll=f["err"].rolling(window, min_periods=5).mean(), roll_raw=f["err_raw"].rolling(window, min_periods=5).mean())
fig2, ax2 = plt.subplots(figsize=(11, 2.8))
ax2.plot(f["day"], f["roll"], label="MAE after EWMA", color="tab:red")
ax2.plot(f["day"], f["roll_raw"], label="MAE raw model", color="tab:gray", lw=0.8)
ax2.axhline(TOL, color="k", ls="--", lw=0.8, label=f"tolerance {TOL}")
ax2.set_xlabel("day"); ax2.set_ylabel(f"rolling MAE ({window})"); ax2.legend(fontsize=7)
st.subheader("실측 웨이퍼 기준 오차 추이")
st.pyplot(fig2)

alarms = f[f["err"] > TOL]
st.subheader(f"허용오차(±{TOL}) 초과 알람: {len(alarms)}건")
st.dataframe(alarms[["day", "wafer_id", "stage", "grp", "actual", "pred", "err", "model_version"]]
             .sort_values("day", ascending=False).head(20).round(2), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 주간 점검과 모델 이력
left, right = st.columns(2)
with left:
    st.subheader("주간 드리프트 점검")
    if events:
        ev = pd.DataFrame([{"주차": e["week"], "운영 모델": e["production"].replace("cmp-xgb-", ""),
                            "실측 MAE(보정 전)": e["mae_raw_week"], "경보": "예" if e["triggered"] else "",
                            "후보 MAE": e.get("mae_candidate"), "운영 MAE": e.get("mae_production"),
                            "승인 요청": "예" if e.get("approval_requested") else ""} for e in events])
        st.dataframe(ev, width="stretch", hide_index=True)
with right:
    st.subheader("모델 교체 이력")
    st.dataframe(pd.DataFrame(registry["history"]), width="stretch", hide_index=True)
