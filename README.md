# 제 5회 ETRI 휴먼이해 인공지능 논문경진대회

## 프로젝트 개요

이 프로젝트는 DACON ETRI 인간이해 대회를 위한 실험용 워크스페이스입니다.  
원천 라이프로그 데이터를 `day-level feature table`로 집계하고, 시간 누수를 막는 검증 폴드를 만든 뒤, `OOF(out-of-fold)` 기준으로 베이스라인 모델을 평가하는 흐름으로 구성되어 있습니다.

현재 파이프라인은 아래 역할을 기준으로 설계되어 있습니다.

- `schema-auditor`: 원천 데이터 스키마, 날짜 정렬, 누수 위험 점검
- `feature-architect`: 일 단위 피처 테이블 설계 및 생성 기준 관리
- `baseline-modeler`: `HGB + smoothed subject prior blend` 중심의 OOF 베이스라인 학습
- `sequence-lite`: 가벼운 단기 시계열 모델 실험
- `validation-paper`: CV, calibration, 실험 카드, 논문화 기준 관리

## 실행 순서

아래 순서대로 실행하면 됩니다.

### 1. 의존성 준비

기본 라이브러리가 없으면 로컬 `.vendor` 경로에 설치합니다.

```bash
python3 -m pip install --target .vendor -r requirements.txt
```

### 2. 스키마 점검

라벨 파일과 parquet 원천 데이터를 읽어 날짜 정렬, 키 구조, 모달리티 스키마를 점검합니다.

```bash
PYTHONPATH=src python3 scripts/run_schema_audit.py
```

생성 결과:

- `artifacts/contracts/schema_contract.json`
- `reports/contracts/schema_contract.md`

### 3. 일 단위 피처 테이블 생성

원천 로그를 `subject_id`, `sleep_date`, `lifelog_date` 기준의 일 단위 피처로 집계합니다.

```bash
PYTHONPATH=src python3 scripts/build_features.py
```

생성 결과:

- `artifacts/features/daily_feature_table.parquet`
- `reports/features/daily_feature_table.md`

### 4. 검증 폴드 생성

`group`과 `group_time` 두 종류의 검증 manifest를 생성합니다.

```bash
PYTHONPATH=src python3 scripts/build_folds.py
```

생성 결과:

- `artifacts/folds/fold_manifest.parquet`
- `reports/folds/fold_manifest.md`

### 5. 베이스라인 학습

먼저 주 평가용 `group_time` split에서 기본 베이스라인 `hgb_prior`를 학습하고, OOF를 확인한 뒤 제출 파일을 생성합니다.  
그다음 필요하면 가드레일용 `group` split과 비교 모델 `catboost`를 추가로 점검합니다.

```bash
PYTHONPATH=src python3 scripts/train_baseline.py --split-scheme group_time --model-family hgb_prior
PYTHONPATH=src python3 scripts/benchmark_models.py
PYTHONPATH=src python3 scripts/make_submission.py --split-scheme group_time --model-family hgb_prior --tag hgb_prior_v1
PYTHONPATH=src python3 scripts/train_baseline.py --split-scheme group --model-family hgb_prior
```

생성 결과:

- `artifacts/oof/oof_predictions_hgb_prior_group_time.parquet`
- `artifacts/oof/oof_predictions_hgb_prior_group.parquet`
- `artifacts/experiments/baseline_hgb_prior_group_time.json`
- `artifacts/experiments/baseline_hgb_prior_group.json`
- `artifacts/submissions/submission_hgb_prior_v1_hgb_prior_group_time.csv`

### 6. Public 트랙 LightGBM 학습

리더보드 최적화용 public 트랙은 notebook 스타일의 `LightGBM + multi-seed StratifiedKFold + transductive subject z-score + target encoding` 흐름으로 별도 운영합니다.  
기존 `daily_feature_table` 위에 `sleep_date 새벽 00~09시` 전용 피처를 추가하고, `public_core` 또는 `public_full` 뷰로 학습합니다.

```bash
PYTHONPATH=src python3 scripts/train_public_lgb.py --feature-view public_core --rebuild-features
PYTHONPATH=src python3 scripts/make_public_lgb_submission.py --feature-view public_core --tag public_lgb_v1
```

생성 결과:

- `artifacts/features/public_lgb_feature_table.parquet`
- `reports/features/public_lgb_feature_table.md`
- `artifacts/oof/oof_predictions_public_lgb_public_core.parquet`
- `artifacts/models/test_predictions_public_lgb_public_core.csv`
- `artifacts/submissions/submission_public_lgb_v1_public_lgb_public_core.csv`

타깃별 블렌딩이 필요하면 두 submission CSV를 직접 섞을 수 있습니다.

```bash
PYTHONPATH=src python3 scripts/mix_submissions.py \
  --left artifacts/submissions/submission_public_lgb_v1_public_lgb_public_core.csv \
  --right artifacts/submissions/submission_v7_s2_w128_public_guarded_group_time.csv \
  --tag public_lgb_s2_mix \
  --target-weight S2=0.35
```

### 7. 선택 사항: 시퀀스 모델 실험

탭уляр 베이스라인보다 나아질 가능성이 있을 때만 경량 시퀀스 모델을 테스트합니다.

```bash
PYTHONPATH=src python3 scripts/train_sequence_lite.py --model-type mlp
PYTHONPATH=src python3 scripts/train_sequence_lite.py --model-type tcn
```

## 핵심 산출물

- `artifacts/features/daily_feature_table.parquet`
  - 학습/추론 전체를 포함한 일 단위 피처 테이블
- `artifacts/folds/fold_manifest.parquet`
  - `group`, `group_time` 검증 분할 정보
- `artifacts/oof/oof_predictions_<model_family>_<scheme>.parquet`
  - 타깃별 `raw`, `cal`, `blend` 확률 예측값
- `artifacts/experiments/*.json`
  - 실험 카드와 점수 기록

## 참고

- `pyarrow`, `catboost` 같은 패키지는 `.vendor`를 먼저 참조하도록 구성되어 있습니다.
- 기본 베이스라인 모델은 `hgb_prior`이며, `HistGradientBoosting + Platt calibration + smoothed subject prior blend`를 사용합니다.
- 실험용 `hgb_select_resid`는 target별 feature selection 후 subject prior residual만 학습합니다. 현재는 기본선 개선이 확인되지 않아 제출 기본값으로 쓰지 않습니다.
- `CatBoost`는 자동 기본값이 아니라 수동 비교용 `--model-family catboost` 옵션으로만 사용합니다.
- 시퀀스 모델은 M1 Pro급 로컬 환경에서 무리 없이 돌릴 수 있는 가벼운 구조만 대상으로 합니다.
- `public_lgb`는 public leaderboard 최적화용 공격적 트랙입니다. `group_time` 기준의 엄격한 재현성과는 분리해서 해석해야 합니다.
