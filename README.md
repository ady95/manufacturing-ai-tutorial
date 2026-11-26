# AI로 만드는 제조공정 품질관리 따라하기 — 예제 코드

위키독스 책 「[AI로 만드는 제조공정 품질관리 따라하기](https://wikidocs.net/book/22288)」(반도체 수율 분석부터 가상계측·AI 불량 판독까지)의 예제 코드 저장소입니다.

모든 Notebook은 **Google Colab**에서 실행하는 것을 기준으로 작성합니다.

## 구성

| 폴더 | 내용 | 데이터셋 | 책 |
|---|---|---|---|
| `01-yield-risk/` | PoC 1. 불량 위험 예측 (이진 분류) | UCI SECOM | 4~6장 |
| `02-virtual-metrology/` | PoC 2. CMP 가상계측 (회귀·시계열) | PHM Society 2016 CMP | 7~10장 |
| `03-defect-inspection/wafermap/` | PoC 3-A. 웨이퍼맵 결함 패턴 분류 | WM-811K | 11~12장 |
| `03-defect-inspection/anomaly-detection/` | PoC 3-B. 제조 이미지 이상 탐지 (PatchCore) | MVTec AD | 13~14장 |
| `04-dashboard/` | 추론 API(FastAPI)와 모니터링 화면(Streamlit) | — | 15~16장 |
| `configs/` | 실험 설정 | | |
| `docs/` | 실험 기록 | | |

각 PoC 폴더는 `notebooks/`(Colab Notebook), `src/`(재사용 코드), `results/`(그래프·비교표)로 나뉩니다.

## 데이터셋

데이터셋 원본은 이 저장소에 포함하지 않습니다. 각 Notebook 첫 부분과 책의 부록 A에서 취득 방법을 안내합니다.

| 데이터셋 | 출처 | 이용 조건 |
|---|---|---|
| UCI SECOM | https://archive.ics.uci.edu/dataset/179/secom | CC BY 4.0 |
| PHM 2016 CMP | https://phmsociety.org/ (2016 PHM Data Challenge). 원본 다운로드 링크가 끊겨 Notebook은 인터넷 아카이브(Wayback Machine) 보존본을 받습니다 | 원본 페이지에 이용 조건 명시 없음. 학습 목적으로 출처를 밝혀 사용 |
| WM-811K | https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map | Kaggle 페이지에 라이선스 표기 없음. 출처를 밝혀 학습 목적으로 사용 |
| MVTec AD | https://www.mvtec.com/research-teaching/datasets/mvtec-ad | CC BY-NC-SA 4.0 (비상업적 이용) |

## 재현 정보

책에 실린 수치는 각 Notebook을 실행해 직접 측정한 값입니다. Notebook마다 실행 일자, 패키지 버전, 랜덤 시드를 기록합니다. 패키지 버전이 바뀌어 결과가 달라지거나 실행되지 않으면 이슈로 알려 주세요.

## 라이선스

코드는 [MIT License](LICENSE)입니다. 단, MVTec AD 이미지가 들어간 결과 그림 4개는 CC BY-NC-SA 4.0(비상업적 이용)을 따르고, 14장에서 쓰는 Ultralytics는 AGPL-3.0입니다. 자세한 예외는 [LICENSE](LICENSE) 파일을 확인하세요.
