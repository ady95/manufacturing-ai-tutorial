# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 14장. 지능형 판독 시스템 고도화
#
# 「AI로 만드는 제조공정 품질관리 따라하기」 14장 실습 Notebook입니다. GPU(T4) 런타임을 쓰세요.
# **13장 Notebook을 먼저 실행해야 합니다**(PatchCore 점수·이상 지도를 불러옵니다).
# Colab에서는 13장 Notebook이 Drive(`MyDrive/manufacturing-ai/13_runs`)에 보관한 결과를 첫 셀이 가져옵니다.
#
# - 데이터: MVTec AD (CC BY-NC-SA 4.0, 비상업적 이용). 결함 마스크를 상자(Bounding Box)로 바꿔 YOLO 라벨로 씁니다
# - YOLO: Ultralytics (AGPL-3.0)
#
# 실험 설계(결과를 보기 전에 정함)
# - 카테고리: bottle, screw, carpet. 13-4와 같은 A/B 분할(평가 이미지를 정상·불량 비율대로 반반)
# - YOLO: A의 이미지(불량은 마스크에서 만든 상자, 정상은 배경)로 학습. yolo26n(COCO 사전학습), 640, 100 epoch 고정,
#   마지막 가중치, 단일 클래스 "defect". 신뢰도 기준 0.25(기본값)
# - PatchCore: 13장 결과(코어셋 10%), 임계값은 A에서 F1 최대
# - 평가: B에서 이미지 AUROC, 기준 적용 시 Recall·오검출, 위치 적중, YOLO 상자 mAP50
# - 새로운 결함 유형: 결함 종류를 하나씩 빼고 YOLO를 다시 학습해, 뺀 종류를 B에서 얼마나 잡는지 PatchCore와 비교
# - 3단계 판정: A에서 두 임계값(미검출 0 / 오검출 0)을 정해 B에 적용 (15개 카테고리, PatchCore)

# %%
import os
import sys
from pathlib import Path

if "google.colab" in sys.modules and not Path("src").exists():
    !git clone -q https://github.com/ady95/manufacturing-ai-tutorial.git
    %cd manufacturing-ai-tutorial/03-defect-inspection/inspection-system
    !pip install -q ultralytics==8.4.218
    from google.colab import drive
    drive.mount("/content/drive")
    TAR = "/content/drive/MyDrive/mvtec_anomaly_detection.tar.xz"      # 13장과 같은 위치
    MVTEC_DIR = "../anomaly-detection/data/mvtec_ad"
    !mkdir -p {MVTEC_DIR} && tar -xJf {TAR} -C {MVTEC_DIR}
    # 13장 Notebook이 Drive에 보관한 PatchCore 결과 (Colab은 Notebook마다 런타임이 따로라서)
    !mkdir -p ../anomaly-detection/results && cp -r /content/drive/MyDrive/manufacturing-ai/13_runs ../anomaly-detection/results/

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
Path("results/14_runs").mkdir(parents=True, exist_ok=True)

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import src.insp_utils as U

ENV = U.env_info()
print(ENV)

# %% [markdown]
# ## 14-1. 마스크에서 상자 라벨 만들기

# %%
label_stats = {}
for c in U.FOCUS:
    s = U.load_pc_scores(c)
    d = s[s.label == 1]
    nbox = [len(U.mask_to_boxes(U.mask_path(c, p))) for p in d["path"]]
    label_stats[c] = {"defect_images": len(d), "boxes": int(sum(nbox)), "max_boxes_per_image": int(max(nbox)),
                      "multi_box_images": int(sum(n > 1 for n in nbox))}
print(label_stats)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, c in zip(axes, U.FOCUS):
    s = U.load_pc_scores(c)
    r = s[s.label == 1].iloc[5]
    img = Image.open(U.MVTEC / c / r["path"]).convert("RGB")
    mask = np.asarray(Image.open(U.mask_path(c, r["path"]))) > 0
    ax.imshow(img); ax.contour(mask, levels=[0.5], colors="yellow", linewidths=1)
    w, h = img.size
    for cx, cy, bw, bh in U.mask_to_boxes(U.mask_path(c, r["path"])):
        ax.add_patch(plt.Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h, fill=False, color="red", lw=1.5))
    ax.set_title(f"{c} | {r['defect_type']} (yellow=mask, red=box)", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig("results/14_mask_to_box.png", dpi=90); plt.show()

# %% [markdown]
# ## YOLO 학습과 평가 (카테고리별 전체 학습 + 결함 종류 하나씩 빼고 학습)
#
# 끝난 학습은 `results/14_runs/`에 저장하고, 다시 실행하면 건너뜁니다.

# %%
U.run_all_jobs_in_subprocesses()                       # 학습은 작업마다 별도 프로세스에서
run_job = U.run_job

full, yolo_b = {}, {}
for c in U.FOCUS:
    full[c], yolo_b[c] = run_job(c)
    print(c, {k: v for k, v in full[c].items() if k not in ("name",)})

# %% [markdown]
# ## 14-2. 같은 B 이미지에서 YOLO와 PatchCore 비교

# %%
rows, hits = [], {}
for c in U.FOCUS:
    s = U.load_pc_scores(c)
    a, b = U.holdout_split(s)
    t_pc = U.f1_threshold(a)
    yb = yolo_b[c].set_index("path").loc[b["path"]].reset_index()
    pc_pred, y_pred = b["score"].to_numpy() >= t_pc, yb["score"].to_numpy() >= U.CONF
    lab = b["label"].to_numpy() == 1
    maps = U.load_pc_maps(c)
    d = b[b.label == 1]
    pc_hit = [U.heatmap_peak_hits_mask(maps[p], U.mask_path(c, p)) for p in d["path"]]
    yb_d = yb[yb.label == 1]
    y_hit = [U.box_hits_mask(bx, U.mask_path(c, p)) for p, bx in zip(yb_d["path"], yb_d["boxes"])]
    for name, score, pred, hit in [("PatchCore", b["score"], pc_pred, pc_hit), ("YOLO", yb["score"], y_pred, y_hit)]:
        rows.append({"category": c, "model": name, "image_AUROC": round(roc_auc_score(lab, score), 4),
                     "recall": round(float(pred[lab].mean()), 3), "fpr": round(float(pred[~lab].mean()), 3),
                     "missed": int((~pred & lab).sum()), "false_alarm": int((pred & ~lab).sum()),
                     "loc_hit": round(float(np.mean(hit)), 3), "B_defect": int(lab.sum()), "B_good": int((~lab).sum())})
compare = pd.DataFrame(rows)
print(compare.to_string(index=False))

# %%
# 결함 종류별 Recall (B): 두 모델이 잘 잡는 종류가 다른가
by_type = []
for c in U.FOCUS:
    s = U.load_pc_scores(c); a, b = U.holdout_split(s); t_pc = U.f1_threshold(a)
    yb = yolo_b[c].set_index("path").loc[b["path"]].reset_index()
    for t in sorted(b.loc[b.label == 1, "defect_type"].unique()):
        m = (b.defect_type == t).to_numpy()
        by_type.append({"category": c, "defect_type": t, "n_B": int(m.sum()),
                        "PatchCore": round(float((b["score"].to_numpy()[m] >= t_pc).mean()), 3),
                        "YOLO": round(float((yb["score"].to_numpy()[m] >= U.CONF).mean()), 3)})
by_type = pd.DataFrame(by_type)
print(by_type.to_string(index=False))

# %% [markdown]
# ### 새로운 결함 유형: 학습에서 뺀 종류를 잡을 수 있는가

# %%
unseen = []
for c in U.FOCUS:
    s = U.load_pc_scores(c); a, b = U.holdout_split(s); t_pc = U.f1_threshold(a)
    for t in sorted(s.loc[s.label == 1, "defect_type"].unique()):
        out, sc = run_job(c, exclude=t)
        m = (sc.defect_type == t).to_numpy()
        pc_m = (b.defect_type == t).to_numpy()
        good = (sc.label == 0).to_numpy()
        unseen.append({"category": c, "held_out": t, "n_B": int(m.sum()),
                       "YOLO_seen": by_type.query("category == @c and defect_type == @t")["YOLO"].item(),
                       "YOLO_unseen": round(float((sc["score"].to_numpy()[m] >= U.CONF).mean()), 3),
                       "YOLO_unseen_fpr": round(float((sc["score"].to_numpy()[good] >= U.CONF).mean()), 3),
                       "PatchCore": round(float((b["score"].to_numpy()[pc_m] >= t_pc).mean()), 3),
                       "train_sec": out["train_sec"]})
        print(unseen[-1])
unseen = pd.DataFrame(unseen)
print(unseen.to_string(index=False))
print(unseen[["YOLO_seen", "YOLO_unseen", "PatchCore"]].mean().round(3))

# %% [markdown]
# ## 14-3. 처리 시간과 3단계 판정

# %%
pc13 = json.loads(Path("../anomaly-detection/results/13_patchcore.json").read_text(encoding="utf-8"))
speed = {"PatchCore 10% (ms/장)": round(float(np.mean([r["ms_10"] for r in pc13["speed"]])), 2),
         "PatchCore 1% (ms/장)": round(float(np.mean([r["ms_1"] for r in pc13["speed"]])), 2),
         "YOLO (ms/장, 전처리+추론+후처리)": round(float(np.mean([sum(full[c]["speed_ms"].values()) for c in U.FOCUS])), 2),
         "YOLO 추론만 (ms/장)": round(float(np.mean([full[c]["speed_ms"]["inference"] for c in U.FOCUS])), 2),
         "YOLO 학습 (초, 평균)": round(float(np.mean([full[c]["train_sec"] for c in U.FOCUS])), 1)}
print(speed)

# %%
three = {}
for c in pd.DataFrame(pc13["table"]).iloc[:, 0]:
    s = U.load_pc_scores(c); a, b = U.holdout_split(s)
    three[c] = U.three_way(a, b)
three_df = pd.DataFrame(three).T
three_df["review_pct"] = (three_df["review"] / three_df["n"] * 100).round(1)
print(three_df.to_string())
tot = three_df[["n", "auto_pass", "review", "auto_ng", "missed_in_pass", "good_in_ng"]].sum()
print("15개 합계:", tot.to_dict(), "| 재검토 비율", round(tot["review"] / tot["n"] * 100, 1), "%")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, c in zip(axes, U.FOCUS):
    s = U.load_pc_scores(c); a, b = U.holdout_split(s); r = three[c]
    bins = np.linspace(s.score.min(), s.score.max(), 30)
    ax.hist(b[b.label == 0].score, bins, alpha=0.6, color="#4C78A8", label="normal (B)")
    ax.hist(b[b.label == 1].score, bins, alpha=0.6, color="#E45756", label="defect (B)")
    ax.axvspan(r["t_low"], r["t_high"], color="gold", alpha=0.3, label="review zone")
    ax.axvline(r["t_low"], color="k", lw=1); ax.axvline(r["t_high"], color="k", lw=1)
    ax.set_title(f"{c}: pass {r['auto_pass']} / review {r['review']} / NG {r['auto_ng']}", fontsize=9)
    ax.set_xlabel("PatchCore score"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/14_three_way.png", dpi=100); plt.show()

# %% [markdown]
# ## 14-4. 판독 화면용 데이터 저장
#
# `app/review_app.py`(Streamlit)가 읽는 파일을 만듭니다.

# %%
queue = []
for c in U.FOCUS:
    s = U.load_pc_scores(c); a, b = U.holdout_split(s); r = three[c]
    yb = yolo_b[c].set_index("path")
    for _, row in b.iterrows():
        decision = "자동 통과" if row.score < r["t_low"] else ("자동 불량" if row.score >= r["t_high"] else "재검토")
        queue.append({"category": c, "path": row.path, "defect_type": row.defect_type, "label": int(row.label),
                      "pc_score": round(float(row.score), 3), "yolo_score": round(float(yb.loc[row.path, "score"]), 3),
                      "yolo_boxes": yb.loc[row.path, "boxes"], "decision": decision})
pd.DataFrame(queue).to_csv("results/14_review_queue.csv", index=False)
Path("results/14_thresholds.json").write_text(json.dumps({c: three[c] for c in U.FOCUS}, indent=2), encoding="utf-8")
print(pd.DataFrame(queue).groupby(["category", "decision"]).size())

# %%
out = {"env": ENV, "yolo_cfg": U.YOLO_CFG, "conf": U.CONF, "label_stats": label_stats,
       "full": full, "compare": compare.to_dict("records"), "by_type": by_type.to_dict("records"),
       "unseen": unseen.to_dict("records"), "unseen_mean": unseen[["YOLO_seen", "YOLO_unseen", "PatchCore"]].mean().round(4).to_dict(),
       "speed": speed, "three_way": three, "three_way_total": tot.astype(int).to_dict()}
Path("results/14_inspection.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
