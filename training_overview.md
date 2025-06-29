# Aardvark 모델 학습 개요

이 문서는 Aardvark 날씨 모델의 학습 데이터 구조와 학습 방법에 대한 개요를 제공합니다.

## 1. 학습 데이터 구조

Aardvark 모델은 다양한 종류의 기상 관측 데이터와 재분석 데이터를 활용하여 학습됩니다. 주요 데이터 소스와 그 역할은 다음과 같습니다.

### 1.1. 주요 데이터 소스

*   **ERA5 재분석 데이터**:
    *   ECMWF에서 제공하는 전 지구 재분석 데이터입니다.
    *   주로 모델의 **실제값(Ground Truth)** 또는 **컨텍스트(Context)** 정보로 사용됩니다. 예를 들어, 특정 시점의 대기 상태를 나타내는 입력으로 사용되거나, 모델이 예측해야 할 목표 상태로 사용됩니다.
    *   `aardvark/loader.py`의 `WeatherDataset` 계열 클래스들에서 `era5_mode` (예: "sfc", "4u", "13u")와 해상도(`res`)에 따라 특정 변수와 레벨이 로드됩니다.
    *   지표 변수(sfc) 또는 특정 기압면 변수들이 활용됩니다.
*   **관측 데이터 (Observational Data)**:
    *   모델의 입력으로 사용되어 현재 대기 상태를 더 정확하게 반영하도록 돕습니다.
    *   `aardvark/loader.py`의 `WeatherDatasetAssimilation` 등에서 다양한 관측 데이터가 로드됩니다.
    *   **HadISD**: 전 지구 지표 관측 데이터 (기온, 이슬점, 해면기압, 바람 등).
    *   **ICOADS**: 해양-대기 관측 데이터.
    *   **IGRA**: 라디오존데(대기 상층 관측) 데이터.
    *   **위성 데이터**:
        *   **AMSU-A/B**: 마이크로파 기반 대기 온도 및 습도 프로파일.
        *   **ASCAT**: 해수면 바람.
        *   **HIRS**: 적외선 기반 대기 온도 및 습도 프로파일.
        *   **GRIDSAT (GEO로 명명)**: 격자화된 위성 데이터.
        *   **IASI**: 적외선 간섭계 기반 고해상도 대기 프로파일.
*   **지형 데이터 (Orography)**:
    *   `era5/elev_vars_{res}.npy` 형태로 저장되며, 모델 입력에 지형 고도 정보로 추가됩니다.
*   **기후학 데이터 (Climatology)**:
    *   `climatology_data.mmap` 형태로 저장되며, 특정 날짜와 시간대의 평균적인 기후 상태를 나타내는 정보로 활용될 수 있습니다.
*   **시간 보조 채널 (Time Auxiliary Channels)**:
    *   연중일(day of year), 연도, 시간(time of day) 정보를 사인/코사인 변환하여 모델 입력으로 사용합니다. 이는 모델이 시간적 주기성을 학습하는 데 도움을 줍니다.

### 1.2. 데이터 형태 및 저장

*   대부분의 대용량 데이터는 빠른 접근을 위해 NumPy 메모리 맵(`.mmap`) 파일 형태로 저장됩니다.
*   각 데이터 파일의 정확한 형태(shape) 정보는 `aardvark/data_shapes.py`에 정의되어 있습니다. 예를 들어, `CLIMATOLOGY_SHAPE`, `ICOADS_Y_SHAPE` 등이 있습니다.
*   `data/sample_data_final.pkl`: 모델 실행 및 예측 데모에 사용되는 샘플 데이터입니다. `notebooks/data_demo.ipynb`에서 이 데이터의 시각화를 확인할 수 있습니다.
*   `data/grid_lon_lat/`: ERA5 데이터의 경도(`era5_x_{res}.npy`) 및 위도(`era5_y_{res}.npy`) 그리드 정보를 담고 있습니다.
*   `data/norm_factors/`: 각 변수 및 데이터 모드에 대한 정규화 인자(평균, 표준편차)가 `.npy` 파일로 저장되어 있습니다. 이는 모델 학습 전 데이터 정규화 및 예측 결과의 역정규화에 사용됩니다. (예: `mean_4u_1.npy`, `std_hadisd_tas.npy`)

### 1.3. 데이터 로더 (`aardvark/loader.py`)

*   `WeatherDataset`을 기본 클래스로 하여 다양한 목적의 데이터셋 클래스들이 정의됩니다.
    *   `WeatherDatasetAssimilation`: 인코더 학습(동화 과정)을 위한 데이터 로더. 다양한 관측 데이터와 ERA5 실제값을 결합합니다.
    *   `ForecastLoader`: 프로세서 모듈 학습/미세조정을 위한 데이터 로더. ERA5 데이터 또는 이전 단계의 모델 예측(초기 조건)을 입력으로 사용합니다.
    *   `WeatherDatasetDownscaling`: 디코더 학습을 위한 데이터 로더.
    *   `WeatherDatasetE2E`: 종단간 학습을 위한 데이터 로더로, 내부적으로 위 데이터셋들을 활용합니다.
*   각 로더는 지정된 기간 및 데이터 종류에 따라 데이터를 로드하고, 정규화하며, 모델 입력에 적합한 형태로 배치(batch)를 구성합니다.
*   `loader_utils_new.py`의 유틸리티 함수(예: 데이터 소스별 오프셋)를 활용합니다.

## 2. 학습 방법

Aardvark 모델은 여러 단계에 걸쳐 학습 및 미세조정됩니다. 각 단계는 특정 모듈의 성능을 최적화하거나 전체 모델의 예측 능력을 향상시키는 것을 목표로 합니다.

### 2.1. 학습 단계 개요

1.  **인코더(Encoder) 학습**:
    *   **목표**: 다양한 관측 데이터를 입력받아 현재 대기 상태에 대한 최적의 분석장(analysis field) 또는 초기 조건(initial condition)을 생성합니다.
    *   **스크립트**: `aardvark/train_module.py` (mode: `assimilation`), `training/train_encoder.sh`
    *   **입력**: 다양한 관측 데이터(HadISD, ICOADS, 위성 등), 지형 데이터, 시간 정보.
    *   **타겟**: 해당 시점의 ERA5 재분석 데이터.
    *   **모델**: `ConvCNPWeather` (주로 `vit_assimilation` 디코더 사용).
2.  **프로세서(Processor) 사전 학습 (Pre-training)**:
    *   **목표**: ERA5 데이터를 사용하여 특정 리드 타임까지의 대기 상태 변화를 예측하는 능력을 학습합니다.
    *   **스크립트**: `aardvark/train_module.py` (mode: `forecast`, ic: `era5`), `training/train_processor.sh`
    *   **입력**: 특정 시점의 ERA5 상태 (초기 조건), 지형, 시간 정보.
    *   **타겟**: 해당 초기 조건으로부터 특정 리드 타임 후의 ERA5 상태.
    *   **모델**: `ConvCNPWeather` (주로 `vit` 디코더 사용).
3.  **프로세서(Processor) 미세 조정 (Fine-tuning)**:
    *   **목표**: 인코더가 생성한 초기 조건을 사용하여, 각 리드 타임별로 프로세서의 예측 성능을 최적화합니다.
    *   **스크립트**: `aardvark/finetune.py`, `training/finetune.sh` (내부적으로 `aardvark/generate_initial_condition_single.py` 호출)
    *   **과정**:
        1.  `generate_initial_condition_single.py`: 학습된 인코더로 초기 조건(`.mmap` 파일) 생성.
        2.  `finetune.py`: 이전 단계에서 학습된 프로세서 모델을 로드하고, 생성된 초기 조건을 입력으로 사용하여 각 리드 타임(1일 ~ 10일)에 대해 순차적으로 미세조정. 이전 리드 타임의 예측 결과가 다음 리드 타임의 입력으로 사용됩니다.
    *   **모델**: `ConvCNPWeather`.
4.  **디코더(Decoder) 학습**:
    *   **목표**: 프로세서가 예측한 저해상도 그리드 결과를 특정 지점(예: 관측소 위치) 또는 고해상도 그리드로 다운스케일링합니다.
    *   **스크립트**: `aardvark/train_module.py` (mode: `downscaling`), `training/train_decoder.sh`
    *   **입력**: 프로세서 예측 결과(Aardvark IC) 또는 ERA5 컨텍스트, 다운스케일링할 목표 지점의 위치 및 고도 정보.
    *   **타겟**: 해당 지점의 실제 관측값(예: HadISD).
    *   **모델**: `ConvCNPWeatherOnToOff` (또는 다른 디코더 아키텍처).
5.  **종단간(End-to-End) 미세 조정**:
    *   **목표**: 인코더, 프로세서, 디코더 모듈을 모두 연결하여 전체 모델의 예측 성능을 특정 변수(예: 기온, 풍속)에 대해 최적화합니다.
    *   **스크립트**: `aardvark/e2e_train.py`, `training/train_e2e.sh`
    *   **입력**: 초기 관측 데이터 (인코더 입력).
    *   **타겟**: 최종 다운스케일링된 지점 예측값.
    *   **모델**: `ConvCNPWeatherE2E` (내부적으로 각 모듈 모델 포함).

### 2.2. 학습 실행

*   `training/` 디렉토리의 쉘 스크립트들은 각 학습 단계를 실행하기 위한 예시 명령어를 제공합니다.
*   각 쉘 스크립트는 주로 `aardvark/train_module.py`, `aardvark/finetune.py`, `aardvark/e2e_train.py` 등의 Python 스크립트를 적절한 인자들과 함께 호출합니다.
*   **주요 학습 스크립트 인자**:
    *   `--output_dir`: 학습 결과(모델 가중치, 로그, 설정 파일 등)가 저장될 경로.
    *   `--mode`: 학습 모드 (`assimilation`, `forecast`, `downscaling`).
    *   `--decoder`: 사용할 ConvCNP 백본 아키텍처 (`vit`, `vit_assimilation`, `base` 등).
    *   `--loss`: 사용할 손실 함수 (`lw_rmse`, `lw_rmse_pressure_weighted`, `rmse`, `downscaling_rmse`).
    *   `--lr`: 학습률.
    *   `--batch_size`: 배치 크기.
    *   `--epoch` 또는 `--finetune_epochs`: 총 에포크 수.
    *   데이터 경로 및 모델 경로 관련 인자들 (예: `--assimilation_model_path`, `--forecast_model_path`).

### 2.3. 학습 시 주의사항

*   **로컬 데이터 로딩 파이프라인**: `README.md`에서 강조하듯이, 제공된 학습 스크립트들은 특정 로컬 데이터 로딩 파이프라인 및 컴퓨팅 인프라에 의존적일 수 있습니다. 따라서 이 저장소의 코드만으로는 모든 학습 과정을 직접 실행하기 어려울 수 있습니다.
*   **경로 설정**: 스크립트 내의 데이터 경로, 모델 저장 경로 등은 실제 사용자 환경에 맞게 수정되어야 합니다.
*   **분산 학습**: 많은 학습 스크립트가 PyTorch의 분산 데이터 병렬 처리(DDP)를 사용하여 여러 GPU에서 학습하는 것을 가정하고 작성되었습니다. 단일 GPU 또는 CPU 환경에서 실행하려면 관련 코드 수정이 필요할 수 있습니다.

이 문서는 Aardvark 모델의 학습 데이터 구조와 학습 방법에 대한 전반적인 이해를 돕기 위해 작성되었습니다. 실제 학습을 진행하기 위해서는 코드의 세부 사항과 각 스크립트의 인자를 면밀히 검토해야 합니다.
