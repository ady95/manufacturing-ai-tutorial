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
# # 7장. 가상계측과 CMP 공정 데이터 이해
#
# 「AI로 만드는 제조공정 품질관리 따라하기」 7장 실습 Notebook입니다.
# Google Colab에서 위에서부터 차례로 실행하면 됩니다.

# %%
# 실행 환경 준비: Colab이면 예제 저장소를 내려받고 실습 폴더로 이동합니다.
import os
import sys
from pathlib import Path

if "google.colab" in sys.modules and not Path("src").exists():
    !git clone -q https://github.com/ady95/manufacturing-ai-tutorial.git
    %cd manufacturing-ai-tutorial/02-virtual-metrology

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
Path("results").mkdir(exist_ok=True)

# %%
import json
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cmp_utils import KEY, add_group, load_cmp

print("Python", platform.python_version(), "| pandas", pd.__version__)

# %% [markdown]
# ## 7-2. 데이터 취득과 파일 구성

# %%
data = load_cmp()          # 처음 한 번 약 11MB를 내려받습니다
for split, (sensors, rr) in data.items():
    print(f"{split:5s} 센서 행 {len(sensors):>7,} | 제거율 {len(rr):>4} | 웨이퍼 {sensors['WAFER_ID'].nunique():>4}")

# %%
raw = Path("data/phm2016/raw")
for split, folder in [("training", "2016 PHM DATA CHALLENGE CMP DATA SET/CMP-data/training"),
                      ("validation", "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET/validation"),
                      ("test", "2016 PHM DATA CHALLENGE CMP DATA SET/CMP-data/test")]:
    files = sorted((raw / folder).glob("*.csv"))
    empty = [f.name for f in files if len(pd.read_csv(f)) == 0]
    print(f"{split:10s} 파일 {len(files)}개 중 헤더만 있는 빈 파일 {len(empty)}개")

# %%
train_s, train_rr = data["train"]
print(train_s.columns.tolist())
train_s.head(3).T

# %%
print("MACHINE_ID 값:", train_s["MACHINE_ID"].unique())
print("MACHINE_DATA와 CHAMBER가 같은 행 비율:", (train_s["MACHINE_DATA"] == train_s["CHAMBER"]).mean())
print("STAGE:", train_s["STAGE"].value_counts().to_dict())

# %%
# 제거율 분포
print(train_rr["AVG_REMOVAL_RATE"].describe().round(2))
print("1000 초과:", (train_rr["AVG_REMOVAL_RATE"] > 1000).sum(), "건")

# %% [markdown]
# ## 7-3. 웨이퍼·단계별 시계열과 실측값 연결

# %%
def summarize(sensors, rr):
    g = sensors.groupby(KEY)
    info = pd.DataFrame({
        "chambers": g["CHAMBER"].agg(lambda s: "".join(str(int(c)) for c in sorted(s.unique()))),
        "t0": g["TIMESTAMP"].min(),
        "duration": g["TIMESTAMP"].max() - g["TIMESTAMP"].min(),
        "n_rows": g.size(),
    }).reset_index()
    out = info.merge(rr, on=KEY, how="outer", indicator=True, validate="one_to_one")
    return add_group(out, sensors)

info = {split: summarize(*v) for split, v in data.items()}
for split, df in info.items():
    print(split, df["_merge"].value_counts().to_dict(), "| 챔버 조합:", df["chambers"].value_counts().to_dict())

# %%
tr = info["train"]
print(tr.groupby("grp")["AVG_REMOVAL_RATE"].describe().round(2))
print("\n이상값(>1000):")
print(tr.loc[tr["AVG_REMOVAL_RATE"] > 1000, ["grp", "AVG_REMOVAL_RATE", "n_rows", "duration"]])

# %%
print("행 수 분포:", tr["n_rows"].describe().round(1).to_dict())
for split, df in info.items():
    long = df[df["n_rows"] > 1000]
    print(split, "행 1000개 초과:", long[["grp", "n_rows", "duration", "AVG_REMOVAL_RATE"]].round(1).values.tolist())

# %%
# 한 웨이퍼가 두 단계를 모두 거친 경우
both = tr[tr["WAFER_ID"].duplicated(keep=False)].pivot(index="WAFER_ID", columns="STAGE", values="t0")
print("두 단계 모두 있는 웨이퍼:", len(both), "| A가 B보다 먼저:", (both["A"] < both["B"]).mean())

# %%
# 세 세트의 관계: 웨이퍼 겹침과 시간 범위
ids = {s: set(df["WAFER_ID"]) for s, df in info.items()}
overlap = {"train-val": len(ids["train"] & ids["val"]),
           "train-test": len(ids["train"] & ids["test"]),
           "val-test": len(ids["val"] & ids["test"])}
print("세트 간 같은 WAFER_ID:", overlap)

allinfo = pd.concat([df.assign(split=s) for s, df in info.items()], ignore_index=True)
t_min = allinfo["t0"].min()
allinfo["day"] = (allinfo["t0"] - t_min) / 86400
for s, df in allinfo.groupby("split"):
    print(f"{s:5s} {df['day'].min():5.1f}일 ~ {df['day'].max():5.1f}일")
print(pd.crosstab((allinfo["day"] // 7).astype(int).rename("week"), allinfo["split"]))

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.4))
colors = {"A123": "#C44E52", "A456": "#4C72B0", "B": "#55A868"}
ok = allinfo[allinfo["AVG_REMOVAL_RATE"] < 1000]
for gname, df in ok.groupby("grp"):
    ax.scatter(df["day"], df["AVG_REMOVAL_RATE"], s=4, alpha=0.5, color=colors[gname], label=gname)
ax.set_xlabel("Day (from first wafer)")
ax.set_ylabel("AVG_REMOVAL_RATE")
ax.set_title("Removal rate over time by process group (all splits)")
ax.legend(markerscale=3, fontsize=8)
fig.tight_layout()
fig.savefig("results/07_rr_over_time.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
for ax, (gname, df) in zip(axes, ok[ok["split"] == "train"].groupby("grp")):
    ax.hist(df["AVG_REMOVAL_RATE"], bins=30, color=colors[gname])
    ax.set_title(f"{gname} (n={len(df)})", fontsize=9)
    ax.set_xlabel("AVG_REMOVAL_RATE", fontsize=8)
fig.suptitle("Training removal rate by process group (outliers > 1000 removed)", fontsize=9)
fig.tight_layout()
fig.savefig("results/07_rr_hist_by_group.png", dpi=150)
plt.show()

# %%
# 시간 추세: 그룹별 하루당 제거율 변화(1차 회귀 기울기)
slope = {g: float(np.polyfit(df["day"], df["AVG_REMOVAL_RATE"], 1)[0]) for g, df in ok.groupby("grp")}
corr_t = {g: float(df["day"].corr(df["AVG_REMOVAL_RATE"])) for g, df in ok.groupby("grp")}
print("하루당 기울기:", {k: round(v, 3) for k, v in slope.items()})
print("시각과의 상관:", {k: round(v, 2) for k, v in corr_t.items()})

# %%
result = {
    "rows": {s: int(len(v[0])) for s, v in data.items()},
    "groups": {s: int(len(v[1])) for s, v in data.items()},
    "wafers": {s: int(v[0]["WAFER_ID"].nunique()) for s, v in data.items()},
    "train_group_counts": tr["grp"].value_counts().to_dict(),
    "train_rr_by_group": tr[tr["AVG_REMOVAL_RATE"] < 1000].groupby("grp")["AVG_REMOVAL_RATE"].describe().round(2).to_dict("index"),
    "outliers_gt_1000": int((tr["AVG_REMOVAL_RATE"] > 1000).sum()),
    "rows_per_group_median": float(tr["n_rows"].median()),
    "duration_median_s": float(tr["duration"].median().round(1)),
    "both_stage_wafers": int(len(both)),
    "wafer_overlap": overlap,
    "days_span": float(allinfo["day"].max().round(1)),
    "slope_per_day": {k: round(v, 3) for k, v in slope.items()},
    "corr_time": {k: round(v, 2) for k, v in corr_t.items()},
}
Path("results/07_eda.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=1))
