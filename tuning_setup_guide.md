# Aardvark 모델 미세조정(Fine-tuning) 가이드

이 문서는 Aardvark 날씨 모델의 프로세서 모듈을 미세조정하기 위한 데이터 준비 및 설정 과정을 안내합니다.

## 개요

Aardvark 모델의 학습은 여러 단계로 진행되며, 그 중 프로세서 모듈은 인코더로부터 생성된 초기 조건(initial conditions)을 입력으로 받아 특정 리드 타임(lead time)까지의 예측을 수행합니다. 이 프로세서 모듈의 성능을 향상시키기 위해 미세조정을 수행할 수 있습니다.

미세조정 과정은 다음과 같습니다:

1.  **초기 조건 생성**: 학습된 인코더 모델을 사용하여 특정 기간(학습, 검증, 테스트 세트)에 대한 초기 조건을 생성합니다. 이 초기 조건은 `.mmap` 파일 형태로 저장됩니다.
2.  **프로세서 모듈 미세조정**: 생성된 초기 조건을 입력으로 사용하여 각 리드 타임별로 프로세서 모듈을 순차적으로 미세조정합니다. 즉, 리드 타임 1에 대해 미세조정한 모델의 예측 결과를 리드 타임 2의 미세조정 입력으로 사용하는 방식입니다.

## 필요 데이터 및 사전 준비

미세조정을 시작하기 전에 다음이 준비되어야 합니다:

1.  **학습된 인코더 모델**: 초기 조건 생성을 위해 미리 학습된 인코더 모델의 경로와 해당 모델의 설정 파일(`config.pkl`)이 필요합니다.
    *   `aardvark/generate_initial_condition_single.py` 스크립트에서 `--encoder_model_path` 인자로 이 경로를 지정합니다.
2.  **사전 학습된 프로세서 모델**: 미세조정의 시작점으로 사용될 사전 학습된 프로세서 모델의 경로와 설정 파일(`config.pkl`)이 필요합니다.
    *   `aardvark/finetune.py` 스크립트에서 `--forecast_model_path` 인자로 이 경로를 지정합니다.
3.  **정규화 인자**: 데이터 정규화 및 역정규화에 사용되는 평균(mean) 및 표준편차(std) 파일이 필요합니다. 이 파일들은 `aux_data_path/norm_factors/` 와 같은 경로에 저장되어 있어야 합니다.
    *   `generate_initial_condition_single.py` 와 `finetune.py` 스크립트 내에서 해당 경로를 참조합니다. (예: `aux_data_path/norm_factors/mean_{era5_mode}_{res}.npy`)
    *   **참고**: 스크립트 내의 `aux_data_path`는 실제 데이터 경로에 맞게 수정해야 할 수 있습니다.
4.  **데이터셋**: `WeatherDatasetAssimilation` 및 `ForecastLoader` 클래스에서 요구하는 형식의 데이터가 준비되어 있어야 합니다. 이는 `loader.py`에 정의된 대로 다양한 기상 관측 데이터(ERA5, HadISD 등)를 포함합니다.
    *   **참고**: `README.md`에 언급된 바와 같이, 전체 데이터셋은 크기 제약으로 인해 이 저장소에 포함되어 있지 않습니다. Hugging Face 저장소([https://huggingface.co/datasets/av555/aardvark-weather](https://huggingface.co/datasets/av555/aardvark-weather))에서 관련 데이터를 다운로드하거나, 로컬 데이터 로딩 파이프라인을 직접 구축해야 할 수 있습니다. 스크립트 내의 데이터 경로는 실제 환경에 맞게 수정되어야 합니다.

## 미세조정 단계별 안내

### 1. 초기 조건 생성 (인코더 예측)

프로세서 모듈 미세조정의 첫 단계는 학습된 인코더를 사용하여 미세조정에 필요한 초기 조건(IC)을 생성하는 것입니다. `aardvark/generate_initial_condition_single.py` 스크립트가 이 작업을 수행합니다.

**실행 예시:**

```bash
python3 ../aardvark/generate_initial_condition_single.py \
    --encoder_model_path ENCODER_MODEL_PATH
```

*   `ENCODER_MODEL_PATH`: 학습된 인코더 모델 가중치와 `config.pkl` 파일이 저장된 디렉토리 경로입니다.
*   이 스크립트는 `train`, `val`, `test` 세트에 대해 지정된 기간 동안 6시간 간격의 초기 조건을 생성하여 `ENCODER_MODEL_PATH` 내에 `ic_train.mmap`, `ic_val.mmap`, `ic_test.mmap` 파일로 저장합니다.
*   생성되는 초기 조건의 형태는 `(시간 스텝 수, 121, 240, 24)`입니다. (위도, 경도, 변수 순서)
*   스크립트 내에서 정규화 인자 경로(`aux_data_path`) 및 데이터 로더(`WeatherDatasetAssimilation`) 관련 경로가 올바르게 설정되어 있는지 확인해야 합니다.

### 2. 프로세서 모듈 미세조정

생성된 초기 조건을 사용하여 각 리드 타임에 대해 프로세서 모듈을 미세조정합니다. `training/finetune.sh` 스크립트는 이 과정을 자동화하며, 내부적으로 `aardvark/finetune.py`를 호출합니다.

**`training/finetune.sh` 스크립트 내용:**

```bash
# 1단계: generate_initial_condition_single.py를 사용하여 초기 조건 생성
# (ENCODER_PATH는 실제 인코더 모델 경로로 대체 필요)
python3 ../aardvark/generate_initial_condition_single.py \
    --assimilation_model_path ENCODER_PATH

# 2단계: finetune.py를 사용하여 프로세서 미세조정
# (ENCODER_PATH, FORECAST_PATH, FINETUNE_PATH는 실제 경로로 대체 필요)
python3 ../aardvark/finetune.py \
    --assimilation_model_path ENCODER_PATH \
    --forecast_model_path FORECAST_PATH \
    --output_dir FINETUNE_PATH \
    --lr 5e-5 \
    --finetune_epochs 1
```

**`aardvark/finetune.py` 스크립트 주요 인자:**

*   `--output_dir`: 미세조정 결과(모델 가중치, 로그, 생성된 예측 데이터 등)가 저장될 디렉토리 경로입니다. (`FINETUNE_PATH`에 해당)
*   `--assimilation_model_path`: 1단계에서 생성된 초기 조건(`.mmap` 파일)이 포함된 디렉토리 경로입니다. (`ENCODER_PATH`에 해당)
*   `--forecast_model_path`: 사전 학습된 프로세서 모델의 경로입니다. (`FORECAST_PATH`에 해당)
*   `--lr`: 학습률 (기본값: 1e-4, `finetune.sh`에서는 5e-5 사용).
*   `--finetune_epochs`: 각 리드 타임별 미세조정 에포크 수 (기본값: 5, `finetune.sh`에서는 1 사용).
*   `--lead_time`: (스크립트 내부에서 1부터 10까지 반복) 현재 미세조정 중인 리드 타임 (일 단위).
*   `--era5_mode`: ERA5 데이터 모드 (기본값: "4u").
*   `--res`: 해상도 (기본값: 1).
*   `--frequency`: 데이터 빈도 (기본값: 6시간).
*   `--batch_size`: 배치 크기 (기본값: 12).
*   `--loss`: 손실 함수 타입 (기본값: "lw_rmse_pressure_weighted").
*   기타 DDP(Distributed Data Parallel) 관련 인자 (`--master_port`).

**미세조정 과정:**

1.  `finetune.py` 스크립트는 지정된 리드 타임(1일부터 10일까지)에 대해 반복적으로 다음을 수행합니다.
2.  **데이터 로딩**:
    *   리드 타임 1의 경우: `--assimilation_model_path` (즉, 인코더 예측)에서 초기 조건을 로드합니다.
    *   리드 타임 > 1의 경우: 이전 리드 타임에서 미세조정된 모델이 생성한 예측 결과 (`{output_dir}/ic_{mode}_{lead_time-1}.mmap`)를 현재 리드 타임의 초기 조건으로 사용합니다.
    *   `ForecastLoader` 데이터셋 클래스를 사용하여 학습, 검증, 테스트 데이터를 준비합니다.
3.  **모델 로딩**: 사전 학습된 프로세서 모델(`--forecast_model_path`)의 가중치를 불러와 현재 리드 타임의 모델을 초기화합니다. (리드 타임 > 1인 경우, 이전 리드 타임에서 학습된 가중치를 사용)
4.  **학습**: `DDPTrainer`를 사용하여 분산 학습 환경에서 모델을 미세조정합니다.
5.  **예측 생성**: 현재 리드 타임에 대해 미세조정된 모델을 사용하여 다음 리드 타임의 학습/검증/테스트 데이터로 사용될 예측값(초기 조건)을 생성하고 `.mmap` 파일로 저장합니다. (예: `FINETUNE_PATH/ic_train_1.mmap`, `FINETUNE_PATH/ic_val_1.mmap` 등)

## 주의사항

*   **경로 설정**: 스크립트 내의 모든 더미 경로 (`ENCODER_PATH`, `FORECAST_PATH`, `FINETUNE_PATH`, `aux_data_path`, `path_to_lat_weights` 등)는 실제 환경에 맞게 올바르게 수정해야 합니다.
*   **데이터 준비**: `README.md`에서 언급된 바와 같이, 모델 학습 및 미세조정에 필요한 전체 데이터셋은 제공되지 않으므로, 사용자가 직접 준비하거나 제공된 샘플 데이터를 기반으로 실험해야 합니다.
*   **실행 환경**: 스크립트들은 분산 학습(DDP)을 가정하고 작성되었으므로, 단일 GPU 또는 CPU 환경에서 실행하려면 수정이 필요할 수 있습니다. `ddp_setup` 및 `DistributedSampler` 관련 부분을 확인해야 합니다.
*   **실행 불가 스크립트**: `README.md`에 명시된 대로, 제공된 학습 스크립트들은 로컬 데이터 로딩 파이프라인 및 특정 컴퓨팅 인프라에 의존적이므로, 이 저장소의 코드만으로는 직접 실행이 어려울 수 있습니다. 이 가이드는 미세조정 과정의 개념과 데이터 흐름을 이해하는 데 중점을 둡니다.

이 가이드가 Aardvark 모델의 미세조정 데이터 설정 과정을 이해하는 데 도움이 되기를 바랍니다.
