# 참고: 이 스크립트는 설명을 위한 예시이며, 전체 데이터셋이 크기 제약으로 인해
# 제출물에 포함되지 않았으므로 직접 실행할 수 없습니다.
# 데이터에 대한 많은 관련 경로는 더미 경로로 대체되었습니다.

import argparse  # 명령줄 인자 파싱을 위한 라이브러리
import pickle  # 파이썬 객체 직렬화 및 역직렬화를 위한 라이브러리

import numpy as np
import pandas as pd
from tqdm import tqdm  # 진행률 표시줄을 위한 라이브러리
import torch
import torch.nn as nn
import torch.utils.data.distributed  # 분산 학습 유틸리티 (여기서는 직접 사용되지 않음)
from torch.utils.data import DataLoader

# 현재 디렉토리의 loader 모듈에서 WeatherDatasetAssimilation 클래스 임포트
from loader import WeatherDatasetAssimilation
# 현재 디렉토리의 models 모듈에서 모든 클래스/함수 임포트
from models import *

# PyTorch의 float32 행렬 곱셈 정밀도 설정 (성능 향상 가능)
torch.set_float32_matmul_precision("medium")


def unnorm(x, mean, std, diff=False, av_2019=None):
    """
    정규화된 데이터를 원래 스케일로 복원(역정규화)합니다.

    Args:
        x (np.ndarray): 정규화된 데이터.
        mean (np.ndarray): 정규화 시 사용된 평균값.
        std (np.ndarray): 정규화 시 사용된 표준편차값.
        diff (bool, optional): 차분(tendency) 데이터인지 여부. Defaults to False.
        av_2019 (np.ndarray, optional): 차분 데이터 복원 시 더해줄 기준값 (예: 2019년 평균).
                                       diff가 True일 때 필요합니다.

    Returns:
        np.ndarray: 역정규화된 데이터.
    """
    x = x * std + mean  # 표준편차를 곱하고 평균을 더하여 스케일 복원
    if diff:
        # 차분 데이터인 경우, 기준값을 더해줌.
        # av_2019의 축 순서 변경 (transpose) 후 더함.
        return x + av_2019.transpose(0, 3, 2, 1)
    return x


if __name__ == "__main__":
    """
    프로세서 모듈의 미세조정 데이터로 사용될 인코더 예측값을 생성합니다.
    즉, 학습된 인코더 모델을 사용하여 특정 기간의 데이터에 대한 초기 조건을 생성하고 저장합니다.
    """

    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_path", help="학습된 인코더 모델이 저장된 경로")
    args = parser.parse_args()

    # 실험 설정 파일 로드 (pickle 파일)
    # 인코더 모델 학습 시 사용된 설정을 불러옴
    with open(args.encoder_model_path + "/config.pkl", "rb") as handle:
        forecast_config = pickle.load(handle)

    device = "cuda"  # GPU 사용 설정

    # 정규화 계수 설정
    era5_mean_spatial = None  # 공간 평균값 (여기서는 사용되지 않지만, unnorm 함수에서 사용될 수 있음)
    # 평균 및 표준편차 로드 (더미 경로 사용)
    means = np.load(
        "aux_data_path/norm_factors/mean_{}_{}.npy".format(  # 실제 경로로 수정 필요
            forecast_config["era5_mode"], forecast_config["res"]
        )
    )[np.newaxis, np.newaxis, np.newaxis, :]  # 배치 및 공간 차원 추가
    stds = np.load(
        "aux_data_path/norm_factors/std_{}_{}.npy".format(  # 실제 경로로 수정 필요
            forecast_config["era5_mode"], forecast_config["res"]
        )
    )[np.newaxis, np.newaxis, np.newaxis, :]

    # 예측값을 생성할 날짜 구간 지정 (학습, 테스트, 검증 세트)
    labels = ["train", "test", "val"]
    dates = [
        ["2007-01-02", "2017-12-31"],  # 학습 데이터 기간
        ["2018-01-01", "2018-12-31"],  # 테스트 데이터 기간
        ["2019-01-01", "2019-12-31"],  # 검증 데이터 기간
    ]

    # 각 데이터 세트(학습, 테스트, 검증)에 대해 반복
    for label, date_period in zip(labels, dates):  # 변수명 date를 date_period로 변경

        # 해당 기간의 시간 스텝 생성 (6시간 간격)
        n_times = pd.date_range(date_period[0], date_period[1], freq="6H")

        # 예측 결과를 저장할 메모리 맵 파일 설정
        # 파일명은 "ic_{label}.mmap" (예: ic_train.mmap)
        ic = np.memmap(
            "{}/ic_{}.mmap".format(args.encoder_model_path, label),
            dtype="float32",
            mode="w+",  # 쓰기 및 읽기 모드, 파일이 없으면 생성
            shape=(len(n_times), 121, 240, 24),  # (시간, 위도, 경도, 변수) 형태 - 확인 필요
        )

        # 변수 그룹별 예측 및 실제값 저장용 리스트 (여기서는 직접 사용되지 않음)
        var_group_preds = []
        var_group_targets = []

        # 데이터 로더 설정
        dataset = WeatherDatasetAssimilation(
            device="cuda",  # GPU 사용
            hadisd_mode="train",  # HadISD 모드 (인코더 학습 시 사용된 모드와 일치해야 할 수 있음)
            start_date=date_period[0],
            end_date=date_period[1],
            lead_time=0,  # 초기 조건 생성이므로 리드 타임 0
            era5_mode="4u",  # ERA5 모드 (설정 파일 값과 일치)
            res=1,  # 해상도 (설정 파일 값과 일치)
            var_start=0,  # 사용할 변수 시작 인덱스
            var_end=24,  # 사용할 변수 끝 인덱스
            diff=False,  # 차분 데이터 사용 안 함
        )

        # DataLoader 생성 (배치 크기 64, 셔플 안 함)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        # 모델 인스턴스화 및 가중치 로드
        model = ConvCNPWeather(  # 모델 클래스 (models.py에 정의되어 있을 것으로 예상)
            in_channels=forecast_config["in_channels"],
            out_channels=forecast_config["out_channels"],
            int_channels=forecast_config["int_channels"],
            device="cuda",
            res=forecast_config["res"],
            gnp=bool(0),  # GNP 사용 여부 (설정 파일 기반)
            decoder=forecast_config["decoder"],  # 디코더 타입 (설정 파일 기반)
            mode=forecast_config["mode"],  # 모델 모드 (설정 파일 기반)
            film=bool(0),  # FiLM 레이어 사용 여부 (설정 파일 기반)
        )

        # 가장 성능이 좋았던 에포크의 가중치 로드
        # 손실값이 저장된 .npy 파일에서 최소 손실을 가진 에포크 번호 찾기
        best_epoch = np.argmin(
            np.load("{}/losses_0.npy".format(args.encoder_model_path)) # losses_0.npy 파일명 확인 필요
        )
        # 해당 에포크의 모델 상태 사전 로드
        state_dict = torch.load(
            "{}/epoch_{}".format(args.encoder_model_path, best_epoch),
            map_location=device,  # 지정된 장치로 로드
        )["model_state_dict"]
        # DataParallel 사용 시 키 이름에 "module." 접두사가 붙으므로 제거
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()} # 키 수정 로직 개선
        model.load_state_dict(state_dict)

        # 여러 GPU 사용을 위한 DataParallel 설정 (단일 GPU 환경에서는 불필요할 수 있음)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device) # 모델을 지정된 장치로 이동 (to(device)가 더 일반적)

        model.eval()  # 모델을 평가 모드로 설정 (dropout, batchnorm 등 비활성화)

        # 예측값 생성
        # total, target 리스트는 여기서는 직접 사용되지 않음
        total = []
        target = []

        sum_count = 0  # 메모리 맵에 저장한 데이터 수 카운트
        with torch.no_grad():  # 그래디언트 계산 비활성화 (추론 시)
            # tqdm을 사용하여 배치 처리 진행률 표시
            with tqdm(loader, unit="batch") as tepoch:
                for count, batch in enumerate(tepoch):
                    # 모델 예측 수행
                    # film_index는 FiLM 레이어 사용 시 특정 조건을 전달하는 데 사용될 수 있음
                    out = model(batch, film_index=batch["lt"]).detach().cpu().numpy()

                    # 예측값 역정규화
                    out_unnorm = unnorm(
                        out,
                        means,
                        stds,
                        diff=False,  # 차분 데이터 아님
                        av_2019=era5_mean_spatial,  # 여기서는 None
                    )

                    # 역정규화된 예측값을 메모리 맵 파일에 저장
                    ic[sum_count : sum_count + out.shape[0], ...] = out_unnorm
                    sum_count += out.shape[0]  # 저장된 데이터 수 업데이트
