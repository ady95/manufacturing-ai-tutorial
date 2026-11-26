"""15-2 실습: 가상계측 추론 API (FastAPI).

실행 (04-dashboard 폴더에서):
    uvicorn app.api:app --port 8000

- GET  /health    운영 모델 버전과 상태
- POST /predict   한 웨이퍼·단계의 원시 센서 기록 → 제거율 예측, 90% 구간, 입력 범위 경고
- POST /feedback  실측값 → 그룹별 EWMA 보정 갱신
- POST /reload    승인된 새 운영 모델로 교체(15-4)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.vm_service import VMService


class SensorRow(BaseModel):
    """원시 센서 기록 한 행. 하나라도 빠지거나 숫자가 아니면 422 오류로 돌려보낸다."""
    TIMESTAMP: float
    CHAMBER: int = Field(ge=1, le=6)
    USAGE_OF_BACKING_FILM: float
    USAGE_OF_DRESSER: float
    USAGE_OF_POLISHING_TABLE: float
    USAGE_OF_DRESSER_TABLE: float
    USAGE_OF_MEMBRANE: float
    USAGE_OF_PRESSURIZED_SHEET: float
    PRESSURIZED_CHAMBER_PRESSURE: float
    MAIN_OUTER_AIR_BAG_PRESSURE: float
    CENTER_AIR_BAG_PRESSURE: float
    RETAINER_RING_PRESSURE: float
    RIPPLE_AIR_BAG_PRESSURE: float
    EDGE_AIR_BAG_PRESSURE: float
    SLURRY_FLOW_LINE_A: float
    SLURRY_FLOW_LINE_B: float
    SLURRY_FLOW_LINE_C: float
    WAFER_ROTATION: float
    STAGE_ROTATION: float
    HEAD_ROTATION: float
    DRESSING_WATER_STATUS: float


class PredictRequest(BaseModel):
    wafer_id: str
    stage: str = Field(pattern="^(A|B)$")
    rows: list[SensorRow] = Field(min_length=10)          # 공정 기록이 너무 짧으면 받지 않음


class FeedbackRequest(BaseModel):
    wafer_id: str
    stage: str = Field(pattern="^(A|B)$")
    actual: float = Field(gt=0, lt=1000)                  # 7장에서 본 4000대 이상값은 거부


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = VMService()                           # 서버가 뜰 때 운영 모델을 한 번 읽는다
    yield


app = FastAPI(title="CMP 가상계측 API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    svc = app.state.svc
    return {"status": "ok", "model_version": svc.version, "cv_mae": svc.meta["cv_mae"],
            "train_n": svc.meta["train_n"]}


@app.post("/predict")
def predict(req: PredictRequest):
    rows = [r.model_dump() for r in req.rows]
    try:
        return app.state.svc.predict(req.wafer_id, req.stage, rows)
    except Exception as e:                                # 특징 계산 실패 등은 500 대신 이유를 알려 준다
        raise HTTPException(status_code=400, detail=f"예측 실패: {e}") from e


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        return app.state.svc.feedback(req.wafer_id, req.stage, req.actual)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/reload")
def reload():
    return {"model_version": app.state.svc.reload()}
