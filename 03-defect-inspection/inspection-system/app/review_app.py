"""14-4 실습: 자동 판독·재검토 화면 (Streamlit).

실행 (inspection-system 폴더에서):
    streamlit run app/review_app.py

14장 Notebook이 만든 results/14_review_queue.csv와 13장의 이상 지도를 읽어,
'재검토'로 분류된 이미지를 엔지니어가 확인하고 판정을 기록합니다. 판정은 results/review_log.csv에 쌓입니다.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
MVTEC = BASE / "../anomaly-detection/data/mvtec_ad"
RUNS13 = BASE / "../anomaly-detection/results/13_runs"            # 13장 Notebook을 직접 돌린 결과
if not (RUNS13 / "bottle_maps.npz").exists():
    RUNS13 = BASE / "../anomaly-detection/results/13_runs_book"   # 없으면 책 실측 결과 (예제 저장소에 포함)
QUEUE = BASE / "results/14_review_queue.csv"
LOG = BASE / "results/review_log.csv"
ORDER = {"재검토": 0, "자동 불량": 1, "자동 통과": 2}

st.set_page_config(page_title="부품 이미지 판독", layout="wide")


@st.cache_data
def load_queue() -> pd.DataFrame:
    q = pd.read_csv(QUEUE)
    q["order"] = q["decision"].map(ORDER)
    return q.sort_values(["order", "pc_score"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data
def load_maps(category: str) -> dict[str, np.ndarray]:
    z = np.load(RUNS13 / f"{category}_maps.npz")
    return {k.replace("__", "/"): z[k].astype(np.float32) for k in z.files}


def load_log() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=["time", "reviewer", "category", "path", "auto_decision", "pc_score",
                                 "yolo_score", "engineer_decision", "note"])


def heatmap_overlay(img: Image.Image, amap: np.ndarray, vmax: float) -> Image.Image:
    import matplotlib

    rgba = matplotlib.colormaps["jet"](np.clip(amap / vmax, 0, 1))
    heat = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).resize(img.size)
    return Image.blend(img, heat, 0.45)


def draw_boxes(img: Image.Image, boxes_json: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for x1, y1, x2, y2, conf in json.loads(boxes_json):
        d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 0, 0), width=3)
        d.text((x1 * w + 3, y1 * h + 3), f"{conf:.2f}", fill=(255, 0, 0))
    return out


queue = load_queue()
log = load_log()

# ---------------------------------------------------------------- 사이드바: 필터와 검토자
st.sidebar.header("판독 대기열")
category = st.sidebar.selectbox("제품(카테고리)", sorted(queue["category"].unique()))
decisions = st.sidebar.multiselect("자동 판정", list(ORDER), default=["재검토"])
reviewer = st.sidebar.text_input("검토자", value="engineer1")
show_truth = st.sidebar.checkbox("정답 보기(실습용)", value=False)

q = queue[(queue["category"] == category) & (queue["decision"].isin(decisions))].reset_index(drop=True)
done = set(log.loc[log["category"] == category, "path"])

# ---------------------------------------------------------------- 상단 요약
cq = queue[queue["category"] == category]
c1, c2, c3, c4 = st.columns(4)
c1.metric("자동 통과", int((cq["decision"] == "자동 통과").sum()))
c2.metric("재검토", int((cq["decision"] == "재검토").sum()))
c3.metric("자동 불량", int((cq["decision"] == "자동 불량").sum()))
c4.metric("판정 기록", f"{len(done & set(cq['path']))} / {len(cq)}")

if q.empty:
    st.info("선택한 조건의 이미지가 없습니다.")
    st.stop()

pending = [i for i, p in enumerate(q["path"]) if p not in done]
idx = st.number_input("이미지 번호", 0, len(q) - 1, value=pending[0] if pending else 0, step=1)
row = q.iloc[int(idx)]

# ---------------------------------------------------------------- 이미지·Heatmap·YOLO 상자
img = Image.open(MVTEC / category / row["path"]).convert("RGB").resize((512, 512))
maps = load_maps(category)
vmax = float(np.percentile(np.concatenate([m.ravel() for m in maps.values()]), 99.5))
a, b, c = st.columns(3)
a.image(img, caption="원본")
b.image(heatmap_overlay(img, maps[row["path"]], vmax), caption=f"PatchCore Heatmap (점수 {row['pc_score']:.2f})")
c.image(draw_boxes(img, row["yolo_boxes"]), caption=f"YOLO 상자 (최고 신뢰도 {row['yolo_score']:.2f})")

st.write(f"**자동 판정: {row['decision']}** · 파일 `{row['path']}`" + (" · 이미 판정함" if row["path"] in done else ""))
if show_truth:
    st.caption(f"정답(실습용): {'불량 - ' + row['defect_type'] if row['label'] else '정상'}")

# ---------------------------------------------------------------- 엔지니어 판정 기록
note = st.text_input("메모", key=f"note_{category}_{idx}")
k1, k2, _ = st.columns([1, 1, 4])
choice = "불량 확정" if k1.button("불량 확정", type="primary") else ("정상 확정" if k2.button("정상 확정") else None)
if choice:
    rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "reviewer": reviewer, "category": category,
           "path": row["path"], "auto_decision": row["decision"], "pc_score": row["pc_score"],
           "yolo_score": row["yolo_score"], "engineer_decision": choice, "note": note}
    pd.concat([log, pd.DataFrame([rec])]).to_csv(LOG, index=False)
    st.success(f"기록했습니다: {row['path']} → {choice}")
    st.rerun()

with st.expander("판정 기록 보기"):
    st.dataframe(load_log().tail(20), width="stretch")
