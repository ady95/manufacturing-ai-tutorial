"""16-2 실습: 세 PoC 통합 모니터링 화면 (Streamlit).

실행 (04-dashboard 폴더에서):
    streamlit run app/integrated.py

세 데이터셋(SECOM, PHM 2016 CMP, MVTec AD)은 서로 다른 출처이며 같은 제조라인의 데이터가 아닙니다.
이 화면은 세 결과를 연결하지 않고, 같은 의사결정 틀로 나란히 보여 줍니다.
필요한 파일은 16장 Notebook(results/16_*), 15장 Notebook(results/vm_log.sqlite), 14장 Notebook이 만듭니다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
TUT = BASE.parent
TOL = 5.0

st.set_page_config(page_title="제조 AI 품질관리 통합 화면", layout="wide")


@st.cache_data
def load_all():
    secom = pd.read_csv(BASE / "results/16_secom_scores.csv", parse_dates=["ts"])
    with sqlite3.connect(BASE / "results/vm_log.sqlite") as con:
        vm = pd.read_sql("SELECT * FROM predictions", con)
        fb = pd.read_sql("SELECT * FROM feedback", con)
    vm["day"] = (vm["t0"] - vm["t0"].min()) / 86400
    fb = fb.merge(vm[["wafer_id", "stage", "day"]].drop_duplicates(["wafer_id", "stage"], keep="last"), on=["wafer_id", "stage"])
    queue = pd.read_csv(TUT / "03-defect-inspection/inspection-system/results/14_review_queue.csv")
    integ = json.loads((BASE / "results/16_integration.json").read_text(encoding="utf-8"))
    return secom, vm, fb, queue, integ


secom, vm, fb, queue, integ = load_all()

st.title("제조 AI 품질관리 통합 화면")
st.caption("세 PoC는 서로 다른 공개 데이터셋입니다. 같은 라인의 데이터가 아니므로 결과를 연결해 해석하지 않습니다.")

tab0, tab1, tab2, tab3 = st.tabs(["공통 의사결정 요약", "불량 위험 예측", "가상계측", "이미지 판독"])

# ---------------------------------------------------------------- 공통 의사결정 요약
with tab0:
    st.subheader("모델 출력 → 현장 행동")
    st.dataframe(pd.DataFrame(integ["summary"]), width="stretch", hide_index=True)
    st.info("비용은 모두 가정값입니다(불량 위험: 놓친 불량 10·헛알람 1 / 가상계측: 실측 1·허용오차 초과 5 / "
            "이미지: 놓친 불량 10·정상 불량 처리 1·재검토 0.2). 실제 비용 절감을 보여 주는 값이 아닙니다.")
    c1, c2, c3 = st.columns(3)
    c1.metric("재검사 알람(불량 위험)", int(secom["alarm"].sum()), help="SECOM 평가 데이터 314건 중")
    c2.metric("추가 계측(가상계측)", len(fb), help="CMP 운영 재현 1,695건 중")
    c3.metric("엔지니어 재검토(이미지)", int((queue["decision"] == "재검토").sum()), help="bottle·screw·carpet B 이미지 중")

# ---------------------------------------------------------------- 불량 위험 예측 (SECOM)
with tab1:
    thr = integ["secom"]["threshold"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평가 웨이퍼", len(secom))
    c2.metric("알람", int(secom["alarm"].sum()))
    c3.metric("알람 중 실제 불량", int((secom["alarm"] & (secom["fail"] == 1)).sum()))
    c4.metric("놓친 불량", int((~secom["alarm"] & (secom["fail"] == 1)).sum()))
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ok, bad = secom[secom["fail"] == 0], secom[secom["fail"] == 1]
    ax.scatter(ok["ts"], ok["risk"], s=8, color="#4C72B0", label="pass")
    ax.scatter(bad["ts"], bad["risk"], s=24, color="#C44E52", marker="x", label="fail")
    ax.axhline(thr, color="k", ls="--", lw=0.8, label=f"alarm threshold {thr}")
    ax.set_ylabel("risk score"); ax.legend(fontsize=7)
    st.pyplot(fig)
    st.warning("이 모델(LR+KBest)의 평가 PR-AUC는 0.135(무작위 기준 0.054)입니다. 알람은 재검사 우선순위를 정하는 참고 정보로만 씁니다(5·6장).")
    st.dataframe(secom[secom["alarm"]].sort_values("risk", ascending=False), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 가상계측 (CMP)
with tab2:
    recent = fb.tail(30)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("운영 모델", vm["model_version"].iloc[-1].replace("cmp-xgb-", ""))
    c2.metric("예측 / 실측", f"{len(vm):,} / {len(fb)}")
    c3.metric("최근 실측 30건 MAE", f"{(recent['actual'] - recent['pred']).abs().mean():.2f}")
    c4.metric("허용오차 초과(실측)", int(((fb["actual"] - fb["pred"]).abs() > TOL).sum()))
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for g, d in vm.groupby("grp"):
        ax.plot(d["day"], d["pred"], lw=0.6, alpha=0.6, label=f"VM {g}")
    ax.scatter(fb["day"], fb["actual"], s=8, color="k", label="measured")
    ax.set_xlabel("day"); ax.set_ylabel("removal rate"); ax.legend(fontsize=7, ncol=4)
    st.pyplot(fig)
    st.caption("자세한 모니터링은 15장 화면(app/monitor.py)을 쓰세요.")

# ---------------------------------------------------------------- 이미지 판독 (MVTec AD)
with tab3:
    counts = queue.pivot_table(index="category", columns="decision", values="path", aggfunc="count", fill_value=0)
    st.subheader("자동 판정 결과(평가 절반 B)")
    st.dataframe(counts, width="stretch")
    st.subheader("재검토 대기열")
    rq = queue[queue["decision"] == "재검토"].sort_values("pc_score", ascending=False)
    st.dataframe(rq[["category", "path", "pc_score", "yolo_score"]], width="stretch", hide_index=True)
    st.caption("이미지를 보고 판정을 기록하려면 14장 화면(03-defect-inspection/inspection-system/app/review_app.py)을 쓰세요.")
