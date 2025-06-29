# 시간 관련 라이브러리 임포트
import time as timelib
from time import time

# NumPy, Pandas, PyTorch 라이브러리 임포트
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 현재 디렉토리의 다른 모듈 임포트
from loader_utils_new import *
from data_shapes import *


class WeatherDataset(Dataset):
    """
    기본 날씨 데이터셋 클래스
    다양한 기상 데이터 소스 (IGRA, AMSU, ICOADS, IASI, GEO, HADISD, ASCAT, HIRS, ERA5)를 로드하고,
    정규화하며, PyTorch Dataset 형태로 제공합니다.
    """

    def __init__(
        self,
        device,  # 데이터가 로드될 장치 (CPU 또는 GPU)
        hadisd_mode,  # HADISD 데이터 모드 (train, val, test 등)
        start_date,  # 데이터 시작 날짜
        end_date,  # 데이터 종료 날짜
        lead_time,  # 예측 리드 타임
        era5_mode="train",  # ERA5 데이터 모드
        res=1,  # 해상도
        filter_dates=None,  # 날짜 필터링 옵션
        diff=None,  # 차분 사용 여부
    ):

        super().__init__()

        # 설정 변수 초기화
        self.device = device
        self.mode = hadisd_mode
        self.data_path = "path_to_data/"  # 주 데이터 경로 (실제 경로로 수정 필요)
        self.aux_data_path = "path_to_auxiliary_data/"  # 보조 데이터 경로 (실제 경로로 수정 필요)
        self.start_date = start_date
        self.end_date = end_date
        self.lead_time = lead_time
        self.era5_mode = era5_mode
        self.res = res
        self.filter_dates = filter_dates
        self.diff = diff

        # 날짜 인덱싱 설정
        self.dates = pd.date_range(start_date, end_date, freq="6H")  # 6시간 간격으로 날짜 범위 생성
        if self.filter_dates == "start":
            # 7월 이전 달만 필터링
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month < 7])
        elif self.filter_dates == "end":
            # 7월 이후 달만 필터링
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month >= 7])
        else:
            # 모든 날짜 사용
            self.index = np.array(range(len(self.dates)))

        # 다양한 입력 데이터 로드
        print("IGRA 데이터 로딩 중")
        self.load_igra()

        print("AMSU-A 데이터 로딩 중")
        self.load_amsua()

        print("AMSU-B 데이터 로딩 중")
        self.load_amsub()

        print("ICOADS 데이터 로딩 중")
        self.load_icoads()

        print("IASI 데이터 로딩 중")
        self.load_iasi()

        print("GEO 위성 데이터 로딩 중")
        self.load_sat_data()  # GRIDSAT 데이터 로드

        print("HADISD 데이터 로딩 중")
        self.load_hadisd(self.mode)

        print("ASCAT 데이터 로딩 중")
        self.load_ascat_data()
        self.load_hirs_data()  # HIRS 데이터 로드

        # 학습을 위한 실제값(ground truth) 데이터 로드
        print("ERA5 데이터 로딩 중")
        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
        ]

        # 내부 그리드와 경도/위도 매핑 정보 로드
        self.era5_x = [
            self.to_tensor(
                np.load(self.data_path + "era5/era5_x_{}.npy".format(self.res))
            )
            / LATLON_SCALE_FACTOR,  # LATLON_SCALE_FACTOR는 경위도 스케일링 상수
            self.to_tensor(
                np.load(self.data_path + "era5/era5_y_{}.npy".format(self.res))
            )
            / LATLON_SCALE_FACTOR,
        ]

        # 지형 데이터(Orography) 로드
        self.era5_elev = self.to_tensor(
            np.load(self.data_path + "era5/elev_vars_{}.npy".format(self.res))
        )
        self.era5_elev = torch.flip(self.era5_elev.permute(0, 2, 1), [-1])  # 축 순서 변경 및 뒤집기
        xx, yy = torch.meshgrid(self.era5_x[0], self.era5_x[1])  # 경위도 그리드 생성
        self.era5_lonlat = torch.stack([xx, yy])  # 경위도 텐서 생성

        # 기후학(Climatology) 데이터 로드 (메모리 맵 사용)
        self.climatology = np.memmap(
            self.data_path + "climatology_data.mmap",
            dtype="float32",
            mode="r",  # 읽기 모드
            shape=CLIMATOLOGY_SHAPE,  # CLIMATOLOGY_SHAPE는 데이터 형태 정의
        )

        # 정규화 계수 설정
        if self.diff:  # 차분 데이터를 사용하는 경우
            self.era5_mean_spatial = np.load(
                self.aux_data_path + "era5_spatial_means.npy"
            )[0, ...]
            self.means = np.load(self.aux_data_path + "era5_avdiff_means.npy")[
                :, np.newaxis, np.newaxis, ...
            ]
            self.stds = np.load(self.aux_data_path + "era5_avdiff_stds.npy")[
                :, np.newaxis, np.newaxis, ...
            ]
        else:  # 원본 데이터를 사용하는 경우
            self.means = np.load(
                self.aux_data_path
                + "norm_factors/mean_{}_{}.npy".format(self.era5_mode, self.res)
            )[:, np.newaxis, np.newaxis, ...]
            self.stds = np.load(
                self.aux_data_path
                + "norm_factors/std_{}_{}.npy".format(self.era5_mode, self.res)
            )[:, np.newaxis, np.newaxis, ...]

    def load_icoads(self):
        """
        ICOADS(International Comprehensive Ocean-Atmosphere Data Set) 데이터를 로드합니다.
        해양-대기 관련 관측 데이터입니다.
        """

        # ICOADS 값 데이터 로드 (메모리 맵)
        self.icoads_y = np.memmap(
            self.data_path + "icoads/1999_2021_icoads_y.mmap",
            dtype="float32",
            mode="r",
            shape=ICOADS_Y_SHAPE,  # ICOADS_Y_SHAPE는 데이터 형태 정의
        )

        # ICOADS 위치 데이터 로드 및 스케일링 (메모리 맵)
        self.icoads_x = (
            np.memmap(
                self.data_path + "icoads/1999_2021_icoads_x.mmap",
                dtype="float32",
                mode="r",
                shape=ICOADS_X_SHAPE,  # ICOADS_X_SHAPE는 데이터 형태 정의
            )
            / LATLON_SCALE_FACTOR
        )
        # ICOADS 평균 및 표준편차 로드 (정규화용)
        self.icoads_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_icoads.npy")
        )
        self.icoads_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_icoads.npy")
        )
        # 최근 1년 데이터 기반으로 평균 및 표준편차 재계산 (NaN 값 고려)
        self.icoads_means = self.to_tensor(
            np.nanmean(self.icoads_y[-365 * 4 :, ...], axis=(0, 2))[:, np.newaxis]
        )
        self.icoads_stds = self.to_tensor(
            np.nanstd(self.icoads_y[-365 * 4 :, ...], axis=(0, 2))[:, np.newaxis]
        )
        # 시작 날짜에 따른 인덱스 오프셋 설정
        self.icoads_index_offset = ICOADS_OFFSETS[self.start_date]  # ICOADS_OFFSETS는 오프셋 딕셔너리
        return

    def load_igra(self):
        """
        IGRA(Integrated Global Radiosonde Archive) 데이터를 로드합니다.
        라디오존데(대기 상층 관측 장비) 데이터입니다.
        """

        # IGRA 값 데이터 로드 (메모리 맵)
        self.igra_y = np.memmap(
            self.data_path + "igra/1999_2021_igra_y.mmap",
            dtype="float32",
            mode="r",
            shape=IGRA_Y_SHAPE,  # IGRA_Y_SHAPE는 데이터 형태 정의
        )

        # IGRA 위치 데이터 로드, 복사 및 스케일링 (메모리 맵)
        self.igra_x = np.copy(
            np.memmap(
                self.data_path + "igra/1999_2021_igra_x.mmap",
                dtype="float32",
                mode="r",
                shape=IGRA_X_SHAPE,  # IGRA_X_SHAPE는 데이터 형태 정의
            )
        )
        self.igra_x = self.igra_x / LATLON_SCALE_FACTOR

        # IGRA 평균 및 표준편차 로드 (정규화용)
        self.igra_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_igra.npy")
        )
        self.igra_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_igra.npy")
        )

        # 시작 날짜에 따른 인덱스 오프셋 설정
        self.igra_index_offset = IGRA_OFFSETS[self.start_date]  # IGRA_OFFSETS는 오프셋 딕셔너리

        return

    def load_amsua(self):
        """
        AMSU-A(Advanced Microwave Sounding Unit-A) 데이터를 로드합니다.
        마이크로파 기반의 대기 온도 관측 위성 데이터입니다.
        """

        # AMSU-A 값 데이터 로드 (메모리 맵)
        self.amsua_y = np.memmap(
            self.data_path + "amsua/2007_2021_amsua.mmap",
            dtype="float32",
            mode="r",
            shape=AMSUA_Y_SHAPE,  # AMSUA_Y_SHAPE는 데이터 형태 정의
        )
        # 시작 날짜에 따른 인덱스 오프셋 설정
        self.amsua_index_offset = AMSUA_OFFSETS[self.start_date]  # AMSUA_OFFSETS는 오프셋 딕셔너리

        # AMSU-A 경도 및 위도 그리드 생성 및 스케일링
        xx = np.linspace(-180, 179, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR  # 0-360도 범위로 변환 후 스케일링
        yy = np.linspace(90, -90, 180, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.amsua_x = [xx, yy]

        # AMSU-A 평균 및 표준편차 로드 (정규화용)
        self.amsua_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_amsua.npy")
        )
        self.amsua_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_amsua.npy")
        )

        return

    def load_amsub(self):
        """
        AMSU-B(Advanced Microwave Sounding Unit-B) 데이터를 로드합니다.
        마이크로파 기반의 대기 습도 관측 위성 데이터입니다.
        """

        # AMSU-B 값 데이터 로드 (메모리 맵)
        self.amsub_y = np.memmap(
            self.data_path + "amsub_mhs/2007_2021_amsub.mmap",  # MHS(Microwave Humidity Sounder) 데이터 포함 가능성
            dtype="float32",
            mode="r",
            shape=AMSUB_Y_SHAPE,  # AMSUB_Y_SHAPE는 데이터 형태 정의
        )
        # 시작 날짜에 따른 인덱스 오프셋 설정
        self.amsub_index_offset = AMSUB_OFFSETS[self.start_date]  # AMSUB_OFFSETS는 오프셋 딕셔너리

        # AMSU-B 경도 및 위도 그리드 생성 및 스케일링
        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(90, -90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.amsub_x = [xx, yy]

        # AMSU-B 평균 및 표준편차 로드 (정규화용)
        self.amsub_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_amsub.npy")
        )
        self.amsub_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_amsub.npy")
        )

        return

    def load_ascat_data(self):
        """
        ASCAT(Advanced Scatterometer) 데이터를 로드합니다.
        해수면 바람 관측 위성 데이터입니다.
        """

        # ASCAT 값 데이터 로드 (메모리 맵)
        self.ascat_y = np.memmap(
            self.data_path + "ascat/2007_2021_ascat.mmap",
            dtype="float32",
            mode="r",
            shape=ASCAT_Y_SHAPE,  # ASCAT_Y_SHAPE는 데이터 형태 정의
        )
        # 시작 날짜에 따른 인덱스 오프셋 설정
        self.ascat_index_offset = ASCAT_OFFSETS[self.start_date]  # ASCAT_OFFSETS는 오프셋 딕셔너리

        # ASCAT 경도 및 위도 그리드 생성 및 스케일링 (위도 순서 반전)
        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.ascat_x = [xx, np.copy(yy[::-1])]  # 위도 배열을 복사하여 뒤집음

        # ASCAT 평균 및 표준편차 로드 (정규화용)
        self.ascat_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_ascat.npy")
        )
        self.ascat_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_ascat.npy")
        )

        return

    def load_hirs_data(self):
        """
        HIRS(High-resolution Infrared Radiation Sounder) 데이터를 로드합니다.
        적외선 기반의 대기 온도 및 습도 관측 위성 데이터입니다.
        """

        # HIRS 값 데이터 로드 (메모리 맵)
        self.hirs_y = np.memmap(
            self.data_path + "hirs/2007_2021_hirs.mmap",
            dtype="float32",
            mode="r",
            shape=HIRS_Y_SHAPE,  # HIRS_Y_SHAPE는 데이터 형태 정의
        )
        # 시작 날짜에 따른 인덱스 오프셋 설정 (ASCAT 오프셋 재활용 가능성 있음)
        self.hirs_index_offset = ASCAT_OFFSETS[self.start_date]

        # HIRS 경도 및 위도 그리드 생성 및 스케일링 (위도 순서 반전)
        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.hirs_x = [xx, np.copy(yy[::-1])]

        # HIRS 평균 및 표준편차 로드 (정규화용)
        self.hirs_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/hirs_means.npy")
        )
        self.hirs_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/hirs_stds.npy")
        )

        return

    def load_sat_data(self):
        """
        GRIDSAT 데이터를 로드합니다.
        격자화된 위성 데이터입니다. (구체적인 위성 종류는 명시되지 않음, GEO로 명명)
        """

        # GRIDSAT 값 데이터 로드 (메모리 맵)
        self.sat_y = np.memmap(
            self.data_path + "gridsat/gridsat_data.mmap",
            dtype="float32",
            mode="r",
            shape=GRIDSAT_Y_SHAPE,  # GRIDSAT_Y_SHAPE는 데이터 형태 정의
        )

        # GRIDSAT 위치 데이터 로드 및 스케일링
        xx = np.load(self.data_path + "gridsat/sat_x.npy") / LATLON_SCALE_FACTOR
        yy = np.load(self.data_path + "gridsat/sat_y.npy") / LATLON_SCALE_FACTOR
        self.sat_x = [xx, yy]
        # 시작 날짜에 따른 인덱스 오프셋 설정
        self.sat_index_offset = SAT_OFFSETS[self.start_date]  # SAT_OFFSETS는 오프셋 딕셔너리

        # GRIDSAT 평균 및 표준편차 로드 (정규화용)
        self.sat_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_sat.npy")
        )
        self.sat_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_sat.npy")
        )

        return

    def load_iasi(self):
        """
        IASI(Infrared Atmospheric Sounding Interferometer) 데이터를 로드합니다.
        적외선 간섭계 기반의 고해상도 대기 관측 위성 데이터입니다.
        """

        # IASI 값 데이터 로드 (메모리 맵, 부분집합일 수 있음)
        self.iasi = np.memmap(
            self.data_path + "2007_2021_iasi_subset.mmap",
            dtype="float32",
            mode="r",
            shape=IASI_Y_SHAPE,  # IASI_Y_SHAPE는 데이터 형태 정의
        )
        # 시작 날짜에 따른 인덱스 오프셋 설정 (ASCAT 오프셋 재활용 가능성 있음)
        self.iasi_index_offset = ASCAT_OFFSETS[self.start_date]

        # IASI 경도 및 위도 그리드 생성 및 스케일링 (위도 순서 반전)
        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.iasi_x = [xx, np.copy(yy[::-1])]

        # IASI 평균 및 표준편차 로드 (정규화용)
        self.iasi_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_iasi.npy")
        )
        self.iasi_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_iasi.npy")
        )

        return

    def load_hadisd(self, mode):
        """
        HadISD(Hadley Centre Integrated Surface Database) 데이터를 로드합니다.
        전 지구 지표 관측 데이터입니다.
        """

        self.hadisd_x = []  # 위치 데이터 (경도, 위도) 리스트
        self.hadisd_alt = []  # 고도 데이터 리스트
        self.hadisd_y = []  # 관측값 데이터 리스트
        hadisd_vars = ["tas", "tds", "psl", "u", "v"]  # 변수: 기온, 이슬점 온도, 해면기압, 동서바람, 남북바람
        for var in hadisd_vars:
            # 경도 데이터 로드 및 0-360 범위로 변환
            lon = lon_to_0_360(  # lon_to_0_360 함수는 경도를 0-360도로 변환
                np.load(
                    self.data_path + "hadisd_processed/{}_lon_{}.npy".format(var, mode)
                )
            )
            # 위도 데이터 로드
            lat = np.load(
                self.data_path + "hadisd_processed/{}_lat_{}.npy".format(var, mode)
            )
            # 고도 데이터 로드
            alt = np.load(
                self.data_path + "hadisd_processed/{}_alt_{}.npy".format(var, mode)
            )

            # 관측값 데이터 로드 (메모리 맵)
            vals = np.memmap(
                self.data_path
                + "hadisd_processed/{}_vals_{}.memmap".format(var, self.mode),
                dtype="float32",
                mode="r",
                shape=get_hadisd_shape(mode),  # get_hadisd_shape 함수는 데이터 형태 반환
            )

            self.hadisd_x.append(np.stack([lon, lat], axis=-1) / LATLON_SCALE_FACTOR)  # 위치 데이터 스케일링 후 추가
            self.hadisd_alt.append(alt)  # 고도 데이터 추가
            self.hadisd_y.append(vals)  # 관측값 데이터 추가

        # 시작 날짜에 따른 인덱스 오프셋 설정
        self.hadisd_index_offset = HADISD_OFFSETS[self.start_date]  # HADISD_OFFSETS는 오프셋 딕셔너리

        # HadISD 변수별 평균 및 표준편차 로드 (정규화용)
        self.hadisd_means = [
            self.to_tensor(
                np.load(
                    self.aux_data_path + "norm_factors/mean_hadisd_{}.npy".format(var)
                )
            )
            for var in hadisd_vars
        ]
        self.hadisd_stds = [
            self.to_tensor(
                np.load(
                    self.aux_data_path + "norm_factors/std_hadisd_{}.npy".format(var)
                )
            )
            for var in hadisd_vars
        ]

        return

    def load_era5(self, year):
        """
        ERA5 재분석 데이터를 로드합니다.
        학습의 실제값(ground truth)으로 주로 사용됩니다.
        """

        # 윤년 여부에 따라 해당 연도의 시간 스텝 수 결정 (6시간 간격이므로 *4)
        if year % 4 == 0:
            d = 366 * 4  # 윤년
        else:
            d = 365 * 4  # 평년

        # ERA5 데이터 모드에 따른 레벨 수 결정
        if self.era5_mode == "sfc":  # 지표면 변수
            levels = 4
        elif self.era5_mode == "13u":  # 13개 연직 레벨 (u 성분 위주)
            levels = 69
        else:  # 기본 24개 레벨
            levels = 24

        # 해상도에 따른 그리드 크기 결정
        if self.res == 1:  # 1도 해상도
            x = 240
            y = 121
        elif self.res == 5:  # 5도 해상도
            x = 64
            y = 32

        # ERA5 데이터 로드 (메모리 맵)
        mmap = np.memmap(
            self.data_path
            + "/era5/era5_{}_{}_6_{}.memmap".format(self.era5_mode, self.res, year),  # 파일명 규칙에 따라 경로 생성
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),  # 데이터 형태 지정
        )
        return mmap

    def norm_era5(self, x):
        """ERA5 데이터를 정규화합니다."""
        x = (x - self.means) / self.stds
        return x

    def unnorm_era5(self, x):
        """정규화된 ERA5 데이터를 원래 스케일로 복원합니다."""
        x = x * self.stds + self.means
        return x

    def norm_data(self, x, means, stds):
        """일반적인 데이터를 주어진 평균과 표준편차로 정규화합니다."""
        return (x - means) / stds

    def norm_hadisd(self, x):
        """HadISD 데이터를 변수별로 정규화합니다."""
        for i in range(5):  # 5개 변수 (tas, tds, psl, u, v)
            x[i] = (x[i] - self.hadisd_means[i]) / self.hadisd_stds[i]
        return x

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        # 마지막 두 샘플은 사용하지 않을 수 있음 (리드 타임 등 고려)
        return self.index.shape[0] - 1 - 1

    def to_tensor(self, arr):
        """NumPy 배열을 PyTorch 텐서로 변환하고 지정된 장치로 옮깁니다."""
        return torch.from_numpy(arr).float().to(self.device)

    def get_time_aux(self, current_date):
        """
        주어진 날짜에 대한 보조 시간 채널(연중일, 연도, 시간)을 생성하여 반환합니다.
        주로 모델 입력으로 사용됩니다.
        """

        doy = current_date.dayofyear  # 연중일 (1~365 또는 366)
        year = (current_date.year - 2007) / 15  # 연도 (2007년을 기준으로 0~1 사이 값으로 정규화)
        time_of_day = current_date.hour  # 시간 (0~23)

        # 삼각함수를 사용하여 주기적인 특성 표현 (cos, sin)
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),  # DAYS_IN_YEAR는 연간 일수 (365.25 등)
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )


class WeatherDatasetAssimilation(WeatherDataset):
    """
    인코더 학습을 위한 데이터 로더 클래스. WeatherDataset을 상속받습니다.
    주로 관측 데이터와 ERA5 실제값을 사용하여 모델의 초기 상태 추정(동화) 단계에 사용됩니다.
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        lead_time,
        era5_mode="sfc",
        res=1,
        filter_dates=None,
        var_start=0,  # 사용할 ERA5 변수의 시작 인덱스
        var_end=24,  # 사용할 ERA5 변수의 끝 인덱스
        diff=False,
        two_frames=False,  # 두 개의 시간 프레임(t=0, t=-1)을 로드할지 여부
    ):

        super().__init__(
            device,
            hadisd_mode,
            start_date,
            end_date,
            lead_time,
            era5_mode,
            res=res,
            filter_dates=filter_dates,
            diff=diff,
        )

        # 설정 변수 초기화
        self.var_start = var_start
        self.var_end = var_end
        self.diff = diff
        self.two_frames = two_frames

    def load_era5_time(self, index):
        """
        특정 시간 인덱스에 해당하는 ERA5 실제값 데이터를 로드하고 정규화합니다.
        """

        date = self.dates[index]
        year = date.year
        hour = date.hour
        doy = (date.dayofyear - 1) * 4 + (hour // 6)  # 연중일 및 시간대 인덱스 계산

        # 해당 연도의 ERA5 데이터에서 특정 시간의 데이터 추출
        era5 = self.era5_sfc[year - int(self.start_date[:4])][doy, ...]
        era5 = np.copy(era5)  # 복사본 사용
        if self.diff:  # 차분 데이터를 사용하는 경우 공간 평균값을 빼줌
            era5 = era5 - self.era5_mean_spatial
        era5 = self.norm_era5(era5[np.newaxis, ...])[0, ...]  # 정규화
        return era5

    def load_year_end(self, year, doy):
        """연말에 걸쳐있는 데이터 로드를 처리합니다 (예: 12월 31일 ~ 1월 1일)."""
        data_1 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]
        missing = self.lead_time - data_1.shape[0] + 1  # 다음 해에서 가져와야 할 데이터 수
        data_2 = self.era5_sfc[year - int(self.start_date[:4]) + 1][:missing, ...]
        data = np.concatenate([data_1, data_2])  # 두 해의 데이터를 합침
        return data

    def load_era5_slice(self, index):
        """
        특정 시간 슬라이스에 대한 ERA5 실제값 데이터를 로드합니다.
        연말 처리를 포함합니다.
        """

        date = self.dates[index]
        year = date.year
        doy = (date.dayofyear - 1) * 4  # 해당 날짜의 첫 번째 시간대 인덱스

        next_date = self.dates[index + 1]
        next_year = next_date.year

        if next_year != year:  # 다음 날짜가 다음 해인 경우
            era5 = self.load_year_end(year, doy)
        else:
            era5 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]

        era5 = self.norm_era5(np.copy(era5))  # 정규화
        return era5

    def __getitem__(self, index):
        """
        주어진 인덱스에 해당하는 데이터 샘플을 반환합니다.
        `two_frames` 옵션에 따라 현재 시간(t=0) 또는 현재와 이전 시간(t=-1) 데이터를 반환합니다.
        """

        if self.two_frames:
            # 경우 1: t=0 및 t=-1 데이터 로드
            index = index + 1  # 인덱스 조정
            current = self.get_index(index, "current")  # 현재 시간 데이터 가져오기
            prev = self.get_index(index - 1, "prev")  # 이전 시간 데이터 가져오기
            current["y_target"] = current["y_target_current"]  # 목표값 설정

            return {**current, **prev}  # 현재 및 이전 데이터 병합하여 반환
        else:
            # 경우 2: t=0 데이터 로드
            current = self.get_index(index, "current")
            current["y_target"] = current["y_target_current"]

            return {**current}

    def unnorm_pred(self, x):
        """정규화된 예측값을 원래 스케일로 복원합니다."""
        dev = x.device  # 현재 장치 가져오기
        x = x.detach().cpu().numpy()  # NumPy 배열로 변환

        # 표준편차와 평균을 사용하여 역정규화
        x = (
            x
            * self.stds[np.newaxis, ...].transpose(0, 2, 3, 1)[  # 축 순서 변경
                ..., self.var_start : self.var_end  # 지정된 변수 범위만 사용
            ]
            + self.means[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
        )
        if bool(self.diff):  # 차분 데이터를 사용한 경우 공간 평균값을 더해줌
            x = (
                x
                + self.era5_mean_spatial[np.newaxis, ...].transpose(0, 3, 2, 1)[
                    ..., self.var_start : self.var_end
                ]
            )
        return torch.from_numpy(x).float().to(dev)  # PyTorch 텐서로 변환 후 장치로 이동

    def get_index(self, index, prefix):
        """
        특정 인덱스에 대한 모든 데이터 소스(ICOADS, GRIDSAT, AMSU 등)의 데이터를 로드하고,
        정규화하며, 딕셔너리 형태로 반환합니다.
        `prefix`는 반환되는 딕셔너리 키의 접두사로 사용됩니다 (예: "current", "prev").
        """

        index = self.index[index]  # 실제 데이터 인덱스로 변환
        date = self.dates[index]  # 해당 날짜 가져오기

        # --- 각 데이터 소스별 로딩 및 전처리 ---
        # ICOADS
        icoads_x_raw = self.icoads_x[index + self.icoads_index_offset, ...]
        icoads_y_raw = self.icoads_y[index + self.icoads_index_offset, ...]
        icoads_x = [icoads_x_raw[0, :], icoads_x_raw[1, :]]  # x, y 좌표 분리
        icoads_x = [self.to_tensor(i) for i in icoads_x]
        icoads_y = self.to_tensor(icoads_y_raw)
        icoads_y = self.norm_data(icoads_y, self.icoads_means, self.icoads_stds)  # 정규화

        # GRIDSAT
        sat_y_raw = self.sat_y[index + self.sat_index_offset, ...]
        sat_x = [self.to_tensor(i) for i in self.sat_x]
        sat_y = self.to_tensor(sat_y_raw)
        sat_y = self.norm_data(sat_y, self.sat_means, self.sat_stds)

        # AMSU-A
        amsua_y_raw = self.to_tensor(self.amsua_y[index + self.amsua_index_offset, ...])
        amsua_y_raw[amsua_y_raw < -998] = torch.nan  # 결측값 처리
        amsua_x = [self.to_tensor(i) for i in self.amsua_x]
        amsua_y = self.norm_data(amsua_y_raw, self.amsua_means, self.amsua_stds)

        # AMSU-B
        amsub_y_raw = self.to_tensor(self.amsub_y[index + self.amsub_index_offset, ...])
        amsub_y_raw[amsub_y_raw < -998] = torch.nan
        amsub_x = [self.to_tensor(i) for i in self.amsub_x]
        amsub_y = self.norm_data(amsub_y_raw, self.amsub_means, self.amsub_stds)

        # IASI
        iasi_y_raw = self.to_tensor(self.iasi[index + self.iasi_index_offset, ...])
        iasi_x = [self.to_tensor(i) for i in self.iasi_x]
        iasi_y = self.norm_data(iasi_y_raw, self.iasi_means, self.iasi_stds)

        # IGRA
        igra_y_raw = self.to_tensor(self.igra_y[index + self.igra_index_offset, ...])
        igra_x_raw = [self.igra_x[:, 0], self.igra_x[:, 1]]
        igra_x = [self.to_tensor(i) for i in igra_x_raw]
        igra_y = self.norm_data(igra_y_raw, self.igra_means, self.igra_stds)

        # ASCAT
        ascat_y_raw = self.to_tensor(self.ascat_y[index + self.ascat_index_offset, ...])
        ascat_x = [self.to_tensor(i) for i in self.ascat_x]
        ascat_y_raw[..., 4][ascat_y_raw[..., 4] < -990] = np.nan  # 특정 채널 결측값 처리
        ascat_y = self.norm_data(ascat_y_raw, self.ascat_means, self.ascat_stds)

        # HIRS
        hirs_y_raw = self.to_tensor(self.hirs_y[index + self.hirs_index_offset, ...])
        hirs_y_raw[hirs_y_raw < -998] = np.nan
        hirs_x = [self.to_tensor(i) for i in self.hirs_x]
        hirs_y = self.norm_data(hirs_y_raw, self.hirs_means, self.hirs_stds)

        # HadISD
        x_context_hadisd_raw = self.hadisd_x
        y_context_hadisd_raw = [
            i[index + self.hadisd_index_offset, :] for i in self.hadisd_y
        ]
        x_context_hadisd = [self.to_tensor(i).permute(1, 0) for i in x_context_hadisd_raw] # 축 순서 변경
        y_context_hadisd_tensor = [self.to_tensor(i) for i in y_context_hadisd_raw]
        y_context_hadisd = self.norm_hadisd(y_context_hadisd_tensor) # 변수별 정규화

        # ERA5 (실제값)
        era5 = self.to_tensor(self.load_era5_time(index))  # 해당 시간의 ERA5 데이터 로드 및 텐서 변환
        era5_target = era5.permute(2, 1, 0)  # 축 순서 변경 (채널, 높이, 너비) -> (너비, 높이, 채널) 예상
        era5_x = self.era5_x  # ERA5 그리드 정보

        # 보조 변수 (시간, 기후학)
        aux_time = self.to_tensor(self.get_time_aux(date))  # 시간 보조 채널 생성
        climatology = self.climatology[date.hour // 6, date.dayofyear - 1, ...]  # 해당 시간 및 연중일의 기후학 데이터

        # 모든 데이터를 딕셔너리 형태로 구성
        task = {
            f"x_context_hadisd_{prefix}": x_context_hadisd,
            f"y_context_hadisd_{prefix}": y_context_hadisd,
            f"climatology_{prefix}": self.to_tensor(climatology),
            f"sat_x_{prefix}": sat_x,
            f"sat_{prefix}": sat_y,
            f"icoads_x_{prefix}": icoads_x,
            f"icoads_{prefix}": icoads_y,
            f"igra_x_{prefix}": igra_x,
            f"igra_{prefix}": igra_y,
            f"amsua_{prefix}": amsua_y,
            f"amsua_x_{prefix}": amsua_x,
            f"amsub_{prefix}": amsub_y,
            f"amsub_x_{prefix}": amsub_x,
            f"iasi_{prefix}": iasi_y,
            f"iasi_x_{prefix}": iasi_x,
            f"ascat_{prefix}": ascat_y,
            f"ascat_x_{prefix}": ascat_x,
            f"hirs_{prefix}": hirs_y,
            f"hirs_x_{prefix}": hirs_x,
            f"y_target_{prefix}": era5_target[  # 목표 ERA5 변수 (지정된 범위)
                ..., self.var_start : self.var_end
            ],
            f"era5_x_{prefix}": era5_x,
            f"era5_elev_{prefix}": self.era5_elev,  # 지형 데이터
            f"era5_lonlat_{prefix}": self.era5_lonlat,  # 경위도 그리드
            f"aux_time_{prefix}": aux_time,
            "lt": torch.Tensor([self.var_start]),  # 리드 타임 (또는 변수 시작 인덱스)
        }

        return task


class HadISDDataset(Dataset):
    """
    디코더 학습을 위한 HadISD 데이터셋 클래스.
    특정 변수(var)에 대한 HadISD 데이터를 로드하고 정규화합니다.
    """

    def __init__(self, var, mode, device, start_date, end_date):
        super().__init__()

        # 설정 변수 초기화
        if not mode in ["train", "val", "test"]:
            raise Exception(f"mode는 {mode}입니다. train, val, test 중 하나여야 합니다.")

        self.var = var  # 대상 변수 (예: 'tas' - 기온)
        self.mode = mode  # 데이터 모드
        self.start_date = start_date
        self.device = device
        dates = pd.date_range(start_date, end_date, freq="6H")
        self.index = np.array(range(len(dates)))

        # HadISD 데이터 로드
        self.load_hadisd()

    def load_hadisd(self):
        """
        원시 HadISD 데이터를 로드합니다.
        """

        data_path = "path_to_data/"  # 주 데이터 경로 (실제 경로로 수정 필요)
        aux_data_path = "path_to_auxiliary_data/"  # 보조 데이터 경로 (실제 경로로 수정 필요)
        var = self.var
        mode = self.mode

        # 관측값 데이터 로드 (메모리 맵)
        vals = np.memmap(
            data_path + f"hadisd_processed/{var}_vals_{mode}.memmap",
            dtype="float32",
            mode="r",
            shape=get_hadisd_shape(mode),
        )

        # 위치 및 고도 데이터 로드
        lon = lon_to_0_360(
            np.load(data_path + f"hadisd_processed/{var}_lon_{mode}.npy")
        )
        lat = np.load(data_path + f"hadisd_processed/{var}_lat_{mode}.npy")
        self.hadisd_x = np.stack([lon, lat], axis=-1) / LATLON_SCALE_FACTOR  # 위치 데이터 스케일링
        self.hadisd_alt = np.load(
            data_path + f"hadisd_processed/{var}_alt_{mode}_final.npy"  # 최종 고도 데이터 사용 가능성
        )
        self.hadisd_y = vals  # 관측값

        # 시작 날짜에 따른 인덱스 오프셋 및 정규화 계수 로드
        self.hadisd_index_offset = HADISD_OFFSETS[self.start_date]
        self.hadisd_means = self.to_tensor(
            np.load(aux_data_path + f"norm_factors/mean_hadisd_{var}.npy")
        )
        self.hadisd_stds = self.to_tensor(
            np.load(aux_data_path + f"norm_factors/std_hadisd_{var}.npy")
        )
        return

    def norm_hadisd(self, x):
        """HadISD 데이터를 정규화합니다."""
        return (x - self.hadisd_means) / self.hadisd_stds

    def unnorm_pred(self, x):
        """정규화된 예측값을 원래 스케일로 복원합니다."""
        return self.hadisd_means + self.hadisd_stds * x

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        return self.index.shape[0] - 2  # 마지막 두 샘플은 사용하지 않을 수 있음

    def to_tensor(self, arr):
        """NumPy 배열을 PyTorch 텐서로 변환합니다."""
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def __getitem__(self, index):
        """주어진 인덱스에 해당하는 데이터 샘플을 반환합니다."""
        index = self.index[index]

        # 목표 위치 (경도, 위도) 가져오기 및 축 변경
        x_target = self.to_tensor(self.hadisd_x).permute(1, 0)

        # 고도 가져오기 및 정규화
        # 고도 평균 및 표준편차 로드 (경로 확인 필요)
        m_alt = np.expand_dims(np.load("path_to_mean_alt.npy"), 1)
        s_alt = np.expand_dims(np.load("path_to_std_alt.npy"), 1)
        alt_target = self.to_tensor((self.hadisd_alt - m_alt) / s_alt)[:, :]

        # 관측값 가져오기 및 정규화
        y_target = self.norm_hadisd(
            self.to_tensor(self.hadisd_y[index + self.hadisd_index_offset, :])
        )

        # 데이터 형태 검증
        assert x_target.shape[0] == 2  # 경도, 위도 2개 채널
        n_stations = x_target.shape[1]  # 관측소 수
        assert alt_target.shape[1] == n_stations
        assert y_target.shape[0] == n_stations

        return {"x": x_target, "altitude": alt_target, "y": y_target}


class AardvarkICDataset(Dataset):
    """
    디코더 학습을 위한 초기 조건(Initial Condition) 로딩 헬퍼 데이터셋.
    Aardvark 모델의 인코더 예측 또는 이전 예측 단계의 결과를 로드합니다.
    """

    def __init__(self, device, start_date, end_date, lead_time=0):
        super().__init__()

        # 설정 변수 초기화
        self.data_path = "path_to_data/" # 데이터 경로 설정 (WeatherDataset과 일치시킬 수 있음)

        if lead_time == 0:
            # 리드 타임이 0이면 인코더의 출력을 로드
            if start_date == "2007-01-02" and end_date == "2017-12-31":
                ic_fname = "ic_train.mmap"  # 학습용 초기 조건 파일
            elif start_date == "2019-01-01" and end_date == "2019-12-01":
                ic_fname = "ic_val.mmap"  # 검증용 초기 조건 파일
            elif start_date == "2018-01-01" and end_date == "2018-12-31":
                ic_fname = "ic_test.mmap"  # 테스트용 초기 조건 파일
            else:
                print((start_date, end_date))
                raise Exception("잘못된 시작 및 종료 날짜입니다.")

            dates = pd.date_range(start_date, end_date, freq="6H")

            # 인코더 예측 데이터 로드 (메모리 맵)
            self.data = np.memmap(
                "path_to_encoder_predictions/" + ic_fname,  # 인코더 예측 저장 경로 (실제 경로로 수정 필요)
                dtype="float32",
                mode="r",
                shape=(len(dates), 121, 240, 24),  # 출력 형태 (시간, 높이, 너비, 채널) - 확인 필요
            )
        else:
            # 리드 타임이 0보다 크면 인코더 예측으로부터 생성된 예측값을 로드
            if start_date == "2007-01-02" and end_date == "2017-12-31":
                ic_fname = f"ic_train_{lead_time}.mmap"
            elif start_date == "2019-01-01" and end_date == "2019-12-01":
                ic_fname = f"ic_val_{lead_time}.mmap"
            elif start_date == "2018-01-01" and end_date == "2018-12-31":
                ic_fname = f"ic_test_{lead_time}.mmap"
            else:
                print((start_date, end_date))
                raise Exception("잘못된 시작 및 종료 날짜입니다.")

            # 리드 타임만큼 날짜 조정
            dates = pd.date_range(start_date, end_date, freq="6H")[(lead_time) * 4 :]
            ic_shape = (len(dates), 121, 240, 24)  # 예측 데이터 형태

            # 예측 데이터 로드 (메모리 맵)
            self.data = np.memmap(
                self.data_path + "forecast_finetune/" + ic_fname,  # 예측 미세조정 데이터 경로 (실제 경로로 수정 필요)
                dtype="float32",
                mode="r",
                shape=ic_shape,
            )

        self.device = device

        # 정규화 계수 로드
        aux_data_path = "path_to_auxiliary_data/"  # 보조 데이터 경로 (실제 경로로 수정 필요)
        mean_factors_path = aux_data_path + f"norm_factors/mean_4u_1.npy"  # 4u 모드, 해상도 1 기준 평균
        std_factors_path = aux_data_path + f"norm_factors/std_4u_1.npy"  # 4u 모드, 해상도 1 기준 표준편차
        self.means = np.load(mean_factors_path)[:, np.newaxis, np.newaxis, ...]
        self.stds = np.load(std_factors_path)[:, np.newaxis, np.newaxis, ...]

    def __getitem__(self, index):
        """주어진 인덱스에 해당하는 Aardvark 예측값을 로드하고 정규화하여 반환합니다."""
        # 데이터 복사 및 축 순서 변경 (채널, 높이, 너비) -> (너비, 높이, 채널) 예상, 확인 필요
        data_raw = np.transpose(np.copy(self.data[index, :, :, :]), (2, 1, 0))
        data = (data_raw - self.means) / self.stds  # 정규화
        return torch.from_numpy(data).to(self.device)


class WeatherDatasetDownscaling(Dataset):
    """
    주요 디코더 학습 데이터셋 클래스.
    AardvarkICDataset (프로세서 출력)과 HadISDDataset (관측소 데이터)을 사용하여
    다운스케일링 모델 학습에 필요한 데이터를 제공합니다.
    """

    def __init__(
        self,
        device,
        hadisd_mode,  # HadISD 데이터 모드
        start_date,
        end_date,
        context_mode,  # 컨텍스트 데이터 모드 ('era5' 또는 'aardvark')
        era5_mode="sfc",
        res=1,
        hadisd_var="tas",  # 사용할 HadISD 변수
        lead_time=1,  # 예측 리드 타임 (일 단위)
    ):
        # 컨텍스트 모드는 ERA5 또는 자체 초기 조건(Aardvark 예측) 사용 여부를 결정
        if not context_mode in ["era5", "aardvark"]:
            raise Exception(
                f"context_mode는 era5 또는 aardvark여야 합니다. 입력값: {context_mode}"
            )

        super().__init__() # Dataset의 __init__ 호출 (명시적으로 호출할 필요는 없음)

        # 설정 변수 초기화
        self.lead_time = lead_time
        self.device = device
        self.data_path = "path_to_data/"
        self.aux_data_path = "path_to_auxiliary_data/"
        self.start_date = start_date
        self.end_date = end_date
        self.era5_mode = era5_mode
        self.res = res
        self.context_mode = context_mode

        self.dates = pd.date_range(start_date, end_date, freq="6H")
        self.index = np.array(range(len(self.dates)))

        # 사전 학습을 위한 ERA5 데이터 로드
        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
        ]

        # ERA5 그리드 정보 로드 및 스케일링
        raw_era5_lon = np.load(self.data_path + f"era5/era5_x_{res}.npy")
        raw_era5_lat = np.load(self.data_path + f"era5/era5_y_{res}.npy")
        self.era5_x = [
            self.to_tensor(raw_era5_lon) / LATLON_SCALE_FACTOR,
            self.to_tensor(raw_era5_lat) / LATLON_SCALE_FACTOR,
        ]

        # 지형 데이터 로드 및 축 변경
        elev_path = self.data_path + f"era5/elev_vars_{res}.npy"
        self.era5_elev = self.to_tensor(np.load(elev_path)).permute(0, 2, 1)

        # 정규화 계수 로드
        mean_factors_path = (
            self.aux_data_path + f"norm_factors/mean_{era5_mode}_{res}.npy"
        )
        std_factors_path = (
            self.aux_data_path + f"norm_factors/std_{era5_mode}_{res}.npy"
        )
        self.means = np.load(mean_factors_path)[:, np.newaxis, np.newaxis, ...]
        self.stds = np.load(std_factors_path)[:, np.newaxis, np.newaxis, ...]

        # HadISD 데이터셋 초기화
        self.hadisd_data = HadISDDataset(
            var=hadisd_var,
            mode=hadisd_mode,
            device=device,
            start_date=start_date,
            end_date=end_date,
        )

        if context_mode == "aardvark":
            # Aardvark 인코더 예측 데이터셋 초기화
            self.aardvark_data = AardvarkICDataset(
                device, start_date, end_date, lead_time
            )

    def load_era5(self, year):
        """원시 ERA5 데이터를 로드합니다. (WeatherDataset의 메서드 재사용)"""
        if year % 4 == 0:
            d = 366 * 4
        else:
            d = 365 * 4

        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        else:
            levels = 24

        if self.res == 1:
            x = 240
            y = 121
        elif self.res == 5:
            x = 64
            y = 32
        mmap = np.memmap(
            self.data_path
            + "era5/era5_{}_{}_6_{}.memmap".format(self.era5_mode, self.res, year),
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def norm_era5(self, x):
        """ERA5 데이터를 정규화합니다."""
        x = (x - self.means) / self.stds
        return x

    def unnorm_era5(self, x):
        """정규화된 ERA5 데이터를 원래 스케일로 복원합니다."""
        x = x * self.stds + self.means
        return x

    def unnorm_pred(self, x):
        """정규화된 예측값을 원래 스케일로 복원합니다. (HadISD 데이터 기준)"""
        return self.hadisd_data.unnorm_pred(x)

    def norm_data(self, x, means, stds):
        """일반적인 데이터를 정규화합니다."""
        return (x - means) / stds

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        # 리드 타임만큼 마지막 샘플들은 사용하지 않음 (6시간 간격이므로 *4)
        return self.index.shape[0] - (self.lead_time) * 4

    def to_tensor(self, arr):
        """NumPy 배열을 PyTorch 텐서로 변환합니다."""
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def get_time_aux(self, current_date):
        """주어진 날짜에 대한 보조 시간 채널을 생성합니다. (WeatherDataset의 메서드 재사용)"""
        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        time_of_day = current_date.hour
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )

    def load_era5_time(self, index):
        """특정 시간 인덱스에 해당하는 ERA5 학습 데이터를 로드합니다."""
        date = self.dates[index]
        year = date.year
        hour = date.hour
        doy = (date.dayofyear - 1) * 4 + (hour // 6)

        era5 = self.era5_sfc[year - int(self.start_date[:4])][doy, ...]
        era5 = np.copy(era5)
        era5 = self.norm_era5(era5[np.newaxis, ...])[0, ...]  # 정규화
        return era5

    def load_year_end(self, year, doy):
        """연말에 걸쳐있는 데이터 로드를 처리합니다. (WeatherDatasetAssimilation의 메서드 재사용)"""
        data_1 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]
        missing = self.lead_time - data_1.shape[0] + 1
        data_2 = self.era5_sfc[year - int(self.start_date[:4]) + 1][:missing, ...]
        data = np.concatenate([data_1, data_2])
        return data

    def __getitem__(self, index):
        """주어진 인덱스에 해당하는 학습 샘플을 반환합니다."""

        index = self.index[index]
        # 리드 타임을 고려한 실제 목표 날짜
        date = self.dates[index + 4 * self.lead_time]

        # HadISD 데이터 가져오기 (목표 시점)
        hadisd_slice = self.hadisd_data[index + 4 * self.lead_time]

        # 컨텍스트 그리드 정보 (경도, 위도)
        x_context = self.era5_x
        n_lon = x_context[0].shape[0]
        n_lat = x_context[1].shape[0]

        # 보조 시간 정보
        aux_time = torch.reshape(self.to_tensor(self.get_time_aux(date)), (-1, 1, 1))

        # 컨텍스트 데이터 로드 (ERA5 또는 Aardvark 예측)
        if self.context_mode == "era5":
            # 사전 학습 시 ERA5 데이터를 컨텍스트로 사용
            y_context_obs = self.to_tensor(
                self.load_era5_time(index + 4 * self.lead_time)
            )
        elif self.context_mode == "aardvark":
            # Aardvark 인코더 예측을 컨텍스트로 사용
            y_context_obs = self.aardvark_data[index]
        else:
            raise Exception("잘못된 context_mode입니다.")

        # 컨텍스트 데이터 구성 (관측/예측값, 지형, 시간 정보 결합)
        y_context = torch.cat(
            [
                y_context_obs,
                self.era5_elev.permute(0, 2, 1),  # 지형 데이터 축 변경
                aux_time.repeat(1, n_lon, n_lat),  # 시간 정보를 그리드 전체에 반복 적용
            ]
        )

        # 데이터 형태 검증
        assert y_context.shape[1] == n_lon
        assert y_context.shape[2] == n_lat

        # HadISD 슬라이스에서 목표값 추출
        x = hadisd_slice["x"]  # 목표 위치
        alt = hadisd_slice["altitude"]  # 목표 고도
        y = hadisd_slice["y"]  # 목표 관측값

        return {
            "x_target": x,
            "alt_target": alt,
            "y_target": y,
            "y_context": y_context,  # 입력 컨텍스트
            "x_context": x_context,  # 입력 컨텍스트 그리드
            "aux_time": aux_time,
            "lt": torch.Tensor([0]),  # 리드 타임 (여기서는 0으로 고정된 듯, 확인 필요)
        }


class ForecasterDatasetDownscaling(Dataset):
    """
    사전 저장된 Aardvark 예측으로부터 디코더 예측을 생성하기 위한 데이터셋.
    주로 미세조정 또는 평가 단계에서 사용될 수 있습니다.
    """

    def __init__(
        self,
        start_date,
        end_date,
        lead_time,  # 예측 리드 타임 (일 단위)
        hadisd_var,  # 사용할 HadISD 변수
        mode,  # 데이터 모드 (train, val, test)
        device,
        forecast_path,  # 사전 저장된 Aardvark 예측 경로 (None일 수 있음)
        region="global",  # 대상 지역 ('global' 또는 특정 지역명)
    ):
        super().__init__()

        # 설정 변수 초기화
        if not mode in ["train", "val", "test"]:
            raise Exception(f"Mode는 {mode}입니다. train, val, test 중 하나여야 합니다.")

        self.device = device
        self.start_date = start_date
        self.end_date = end_date
        self.lead_time = lead_time
        self.mode = mode
        self.data_path = "path_to_data/" # 데이터 경로 추가 (WeatherDatasetDownscaling과 일치시킬 수 있음)
        # 리드 타임만큼 시간 오프셋 설정
        self.offset = np.timedelta64(lead_time, "D").astype("timedelta64[ns]")

        # 날짜 범위 생성 (마지막 30개는 제외하는 것으로 보임, 확인 필요)
        self.dates = pd.date_range(start_date, end_date, freq="6H")[:-30]

        # 정규화 계수 로드 (4u 모드, 해상도 1 기준)
        aux_data_path = "auxiliary_data_path/"  # 실제 경로로 수정 필요
        self.means = np.load(aux_data_path + "norm_factors/mean_4u_1.npy")
        self.stds = np.load(aux_data_path + "norm_factors/std_4u_1.npy")

        # 보조 데이터 로드 (ERA5 그리드, 지형)
        self.load_npy_file() # 사전 저장된 예측 로드 (forecast_path가 None이 아니면)
        # data_path = "data_path/" # 중복된 선언, 클래스 멤버 변수로 사용
        res = "1" # 해상도 고정
        raw_era5_lon = np.load(self.data_path + f"era5/era5_x_{res}.npy")
        raw_era5_lat = np.load(self.data_path + f"era5/era5_y_{res}.npy")
        self.era5_x = [
            self.to_tensor(raw_era5_lon) / LATLON_SCALE_FACTOR,
            self.to_tensor(raw_era5_lat) / LATLON_SCALE_FACTOR,
        ]
        elev_path = self.data_path + f"era5/elev_vars_{res}.npy"
        self.era5_elev = self.to_tensor(np.load(elev_path)).permute(0, 2, 1)

        # HadISD 데이터셋 초기화 (여기서는 mode='train'으로 고정, 확인 필요)
        self.hadisd_data = HadISDDataset(
            var=hadisd_var,
            mode="train", # mode 변수를 사용하는 것이 더 일반적일 수 있음
            device=device,
            start_date=start_date,
            end_date=end_date,
        )

        # 지역별 마스크 로드 (전역이 아닌 경우)
        self.region = region
        if self.region != "global":
            self.mask = np.load(
                self.data_path + f"hadisd_processed/tas_mask_train_{region}.npy" # tas 변수, train 모드 마스크 고정
            )

    def date_range(self):
        """시작일부터 종료일까지 1일 간격의 날짜 범위를 생성합니다."""
        return np.arange(
            start=np.datetime64(self.start_date).astype("timedelta64[ns]"),
            stop=np.datetime64(self.end_date).astype("timedelta64[ns]"),
            step=np.timedelta64(1, "D").astype("timedelta64[ns]"),
        )

    def load_npy_file(self):
        """
        사전 저장된 Aardvark 예측(.mmap 파일)을 로드합니다.
        forecast_path가 설정되어 있어야 합니다. (현재 코드에서는 forecast_path를 직접 사용하지 않고 고정된 경로 사용)
        """

        dates = pd.date_range(self.start_date, self.end_date, freq="6H")

        if self.mode == "train":
            dates = dates[:-40]  # 학습 모드일 때 마지막 40개 제외 (10일 오프셋)

        # 예측 데이터 로드 (메모리 맵, 경로 확인 필요)
        self.Y_context = np.memmap(
            "path_to_forecasts/forecast_{}.mmap".format(self.mode),  # 실제 예측 저장 경로로 수정 필요
            dtype="float32",
            mode="r",
            shape=(len(dates), 121, 240, 24, 11),  # (시간, 높이, 너비, 채널, 리드타임_단계) - 확인 필요
        )
        return

    def norm_era5(self, x):
        """ERA5 데이터를 정규화합니다."""
        return (x - self.means[:, np.newaxis, np.newaxis, :self.Y_context.shape[-2]]) / self.stds[:, np.newaxis, np.newaxis, :self.Y_context.shape[-2]] # Y_context의 채널 수에 맞게 슬라이싱

    def norm_hadisd(self, x):
        """HadISD 데이터를 정규화합니다."""
        return self.hadisd_data.norm_hadisd(x)

    def unnorm_pred(self, x):
        """정규화된 예측값을 원래 스케일로 복원합니다. (HadISD 데이터 기준)"""
        return self.hadisd_data.unnorm_pred(x)

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        return len(self.dates) - 40  # 마지막 40개 제외 (10일 오프셋)

    def to_tensor(self, arr):
        """NumPy 배열을 PyTorch 텐서로 변환합니다."""
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def get_time_aux(self, index):
        """주어진 인덱스에 대한 보조 시간 변수를 가져옵니다."""
        # 리드 타임을 고려한 실제 날짜
        current_date = (self.dates + self.offset)[index]
        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        time_of_day = current_date.hour
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )

    def __getitem__(self, index):
        """주어진 인덱스에 해당하는 학습 샘플을 반환합니다."""

        # 목표 데이터 로드 (HadISD)
        hadisd_slice = self.hadisd_data[index + 4 * self.lead_time] # 리드 타임(일)을 6시간 간격으로 변환 (*4)

        # 컨텍스트 그리드 정보
        x_context = self.era5_x
        n_lon = x_context[0].shape[0]
        n_lat = x_context[1].shape[0]

        # 보조 시간 정보
        aux_time = torch.reshape(self.to_tensor(self.get_time_aux(index)), (-1, 1, 1))

        # 입력 컨텍스트 로드 (사전 저장된 Aardvark 예측)
        # self.Y_context의 마지막 차원은 리드타임 단계로 보임, self.lead_time (일 단위) 사용 방식 확인 필요
        # 여기서는 self.lead_time을 인덱스로 직접 사용
        y_context_raw = self.Y_context[index, ..., self.lead_time]
        y_context_norm = self.norm_era5(y_context_raw) # 정규화
        y_context = torch.cat(
            [
                self.to_tensor(y_context_norm).permute(2, 1, 0), # 축 변경 (채널, 높이, 너비) -> (너비, 높이, 채널) 예상
                self.era5_elev.permute(0, 2, 1),
                aux_time.repeat(1, n_lon, n_lat),
            ]
        )

        # 데이터 형태 검증
        assert y_context.shape[1] == n_lon
        assert y_context.shape[2] == n_lat

        # 지역 마스킹 처리
        if self.region != "global":
            hadisd_slice["y"][self.mask] = np.nan  # 마스크된 지역의 목표값을 NaN으로 설정

        return {
            "x_target": hadisd_slice["x"],
            "alt_target": hadisd_slice["altitude"],
            "y_target": hadisd_slice["y"],
            "y_context": y_context,
            "x_context": x_context,
            "aux_time": aux_time,
            "lt": torch.Tensor([0]), # 리드 타임 (여기서는 0으로 고정된 듯, 확인 필요)
        }


class ForecastLoader(Dataset):
    """
    프로세서 모듈 미세조정을 위한 로더.
    ERA5 데이터를 기반으로 예측 모델의 입력과 목표값을 생성합니다.
    초기 조건(ic_path)을 사용하거나 ERA5 자체를 초기 조건으로 사용할 수 있습니다.
    """

    def __init__(
        self,
        device,
        mode,  # 데이터 모드 (train, tune, test, val)
        lead_time,  # 예측 리드 타임 (시간 또는 일 단위, frequency에 따라 해석)
        era5_mode="sfc",
        res=5,  # 해상도
        frequency=24,  # 데이터 빈도 (6시간 또는 24시간)
        norm=True,  # 정규화 사용 여부
        diff=False,  # 차분(tendency) 사용 여부
        rollout=False,  # 전체 타임 시리즈 반환 여부
        random_lt=False,  # 랜덤 리드 타임 오프셋 사용 여부
        u_only=False,  # u 성분만 사용하는지 (현재 코드에서는 직접 사용되지 않음)
        ic_path=None,  # 초기 조건(.mmap 파일) 경로
        finetune_step=None,  # 미세조정 단계 (이전 단계의 예측을 초기 조건으로 사용)
        finetune_eval_every=100, # 미세조정 중 평가 빈도
        eval_steps=False, # 평가 단계인지 여부
    ):

        super().__init__()

        # 설정 변수 초기화
        self.device = device
        self.mode = mode
        self.data_path = "data_path/" # 실제 경로로 수정 필요

        self.lead_time = lead_time
        self.era5_mode = era5_mode
        self.res = res
        self.frequency = frequency
        self.norm = norm
        self.diff = diff
        self.rollout = rollout
        self.random_lt = random_lt
        self.u_only = u_only # 사용되지 않는 변수
        self.ic_path = ic_path

        self.finetune_step = finetune_step
        self.finetune_eval_every = finetune_eval_every
        self.eval_steps = eval_steps

        # 빈도에 따라 리드 타임 조정 및 날짜 생성 빈도 설정
        if self.frequency == 6: # 6시간 빈도
            self.lead_time = self.lead_time * 4 # 리드 타임을 6시간 단위로 변환
            freq_str = "6H"
        else: # 24시간 빈도 (기본값)
            freq_str = "1D"

        # 모드에 따른 날짜 범위 설정
        if self.mode == "train":
            self.dates = pd.date_range("1979-01-01", "2017-12-31", freq=freq_str)
        elif self.mode == "tune": # 미세조정용 데이터
            self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq_str)
        elif self.mode == "test":
            self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq_str)
        elif self.mode == "val":
            self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq_str)

        # 미세조정 단계에 따른 초기 조건 로드
        if self.finetune_step is not None:
            # 날짜 범위 재설정 (Aardvark 학습 기간과 유사하게)
            if self.mode == "train":
                self.dates = pd.date_range("2007-01-02", "2017-12-31", freq=freq_str)
            elif self.mode == "val":
                self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq_str)
            elif self.mode == "test":
                self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq_str)

            # 초기 조건 데이터 형태 계산
            ic_shape_len_offset = max(0, (self.finetune_step - 1) * (4 if self.frequency == 6 else 1)) # 이전 단계 예측 사용 시 길이 조정
            ic_shape = (len(self.dates) - ic_shape_len_offset, 121, 240, 24) # (시간, 높이, 너비, 채널) - 확인 필요

            if self.finetune_step > 1: # 두 번째 미세조정 단계 이후
                print(ic_shape)
                self.ic = np.memmap(
                    self.ic_path + f"ic_{self.mode}_{self.finetune_step - 1}.mmap", # 이전 단계 예측 파일
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )
            elif self.ic_path is not None: # 첫 번째 미세조정 단계 또는 일반 초기 조건 사용
                self.ic = np.memmap(
                    self.ic_path + f"ic_{self.mode}.mmap", # 기본 초기 조건 파일
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )
        elif self.ic_path is not None: # 일반 초기 조건 사용 (미세조정 단계 아님)
            if self.mode == "train": # 학습 모드 시 날짜 범위 재설정
                self.dates = pd.date_range("2007-01-02", "2017-12-31", freq=freq_str)
            ic_shape = (len(self.dates), 121, 240, 24)
            self.ic = np.memmap(
                self.ic_path + f"/ic_{self.mode}.mmap",
                dtype="float32",
                mode="r",
                shape=ic_shape,
            )

        # 지형 데이터 로드 및 정규화
        self.era5_elev = np.float32(
            np.load(self.data_path + "era5/elev_vars_{}.npy".format(res))
        )
        elev_mean = self.era5_elev.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
        elev_std = self.era5_elev.std(axis=(1, 2))[:, np.newaxis, np.newaxis]
        self.era5_elev = (self.era5_elev - elev_mean) / elev_std

        # 학습을 위한 ERA5 실제값 데이터 로드
        self.era5_sfc = [
            self.load_era5(year) # 연도별 ERA5 데이터 로드
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]

        # 정규화 계수 로드 (원본 데이터용 및 차분 데이터용)
        self.means = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_{}_{}.npy".format(self.era5_mode, self.res)
                )
            )
            .unsqueeze(1) # 차원 확장
            .unsqueeze(1)
        )
        self.stds = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_{}_{}.npy".format(self.era5_mode, self.res)
                )
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        # 차분(tendency) 데이터용 정규화 계수 (다른 리드 타임 오프셋에 대한 계수들)
        self.diff_means = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_diff_{}_{}.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.diff_stds = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_diff_{}_{}.npy".format(self.era5_mode, self.res)
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # 6시간 후 차분 데이터용
        self.diff_means_1 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_diff_{}_{}_6h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.diff_stds_1 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_diff_{}_{}_6h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # 12시간 후 차분 데이터용
        self.diff_means_2 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_diff_{}_{}_12h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.diff_stds_2 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_diff_{}_{}_12h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # 리드 타임 오프셋에 따른 정규화 계수 딕셔너리
        self.means_dict = {
            0: self.diff_means,  # 기본 차분
            2: self.diff_means_2, # 12시간 후 차분 (lt_offset = 2 -> 12시간, frequency=6H 가정)
            3: self.diff_means_1,  # 6시간 후 차분 (lt_offset = 3 -> 6시간, frequency=6H 가정, 확인 필요, 1이 더 적절해 보임)
        }
        self.stds_dict = {0: self.diff_stds, 2: self.diff_stds_2, 3: self.diff_stds_1}

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        if np.logical_and(self.eval_steps, self.mode == "train"): # 학습 중 평가 단계인 경우
            # 평가 빈도에 맞춰 샘플 수 제한 (12는 특정 설정값으로 보임, 4는 6H 빈도에서 하루의 스텝 수)
            return self.finetune_eval_every * 12 * (4 if self.frequency == 6 else 1)
        # 리드 타임만큼 마지막 샘플들은 사용하지 않음
        return self.dates.shape[0] - self.lead_time

    def to_tensor(self, arr):
        """NumPy 배열을 PyTorch 텐서로 변환합니다."""
        return torch.from_numpy(arr).float().to(self.device)

    def norm_era5(self, x, lt_offset=None): # lt_offset 파라미터 추가 (WeatherDatasetE2E에서 호출 시 사용)
        """ERA5 데이터를 정규화합니다."""
        # lt_offset는 이 클래스 내에서는 현재 사용되지 않음. diff=True일 때 norm_era5_tendency 사용
        x = (x - self.means) / self.stds
        return x

    def norm_era5_tendency(self, x, lt_offset):
        """ERA5 차분(tendency) 데이터를 정규화합니다."""
        x = (x - self.means_dict[lt_offset]) / self.stds_dict[lt_offset]
        return x

    def unnorm_pred(self, x):
        """정규화된 차분 예측값을 원래 스케일로 복원합니다."""
        # 차분 예측이므로 차분 평균/표준편차 사용
        x = x * self.diff_stds.unsqueeze(0) + self.diff_means.unsqueeze(0)
        return x

    def unnorm_base_context(self, x):
        """정규화된 기본 컨텍스트(초기 조건)를 원래 스케일로 복원합니다."""
        x = x * self.stds.unsqueeze(0) + self.means.unsqueeze(0)
        return x

    def load_era5(self, year):
        """특정 연도의 ERA5 데이터를 로드합니다."""
        # 윤년 여부 및 데이터 빈도에 따른 해당 연도의 시간 스텝 수 결정
        if year % 4 == 0:
            d = 366
        else:
            d = 365
        if self.frequency == 6:
            d = d * 4

        # ERA5 모드 및 해상도에 따른 레벨 및 그리드 크기 결정
        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        else:
            levels = 24

        if self.res == 1:
            x = 240
            y = 121
        elif self.res == 5:
            x = 64
            y = 32

        # ERA5 데이터 로드 (메모리 맵)
        mmap = np.memmap(
            self.data_path
            + "era5/era5_{}_{}_{}_{}.memmap".format( # 파일명 규칙에 빈도(frequency) 포함
                self.era5_mode, self.res, self.frequency, year
            ),
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def load_era5_time(self, index):
        """특정 시간 인덱스에 해당하는 ERA5 데이터를 로드합니다."""
        date = self.dates[index]
        year = date.year
        doy = date.dayofyear - 1  # 연중일 (0부터 시작)
        hour = date.hour

        # 빈도에 따라 해당 연도의 ERA5 데이터에서 특정 시간의 데이터 추출
        if self.frequency == 6: # 6시간 빈도
            era5 = self.era5_sfc[year - int(self.dates[0].year)][ # 해당 연도 데이터 접근
                doy * 4 + hour // 6, ... # 시간대 인덱스 계산
            ]
        else: # 24시간 빈도
            era5 = self.era5_sfc[year - int(self.dates[0].year)][doy, ...]
        return np.copy(era5)

    def make_time_channels(self, index, x, y):
        """주어진 인덱스에 대한 보조 시간 채널을 생성합니다."""
        date = self.dates[index]
        hour = date.hour
        doy = date.dayofyear - 1
        if date.year % 4 == 0: # 윤년 처리
            n_days = 366
        else:
            n_days = 365

        # 시간 및 연중일을 sin/cos 함수로 변환하여 주기적 특성 표현
        hour_sin = np.sin(hour * np.pi / 12) * np.float32(np.ones((1, x, y))) # 12시간 주기 가정 (pi/12) - 확인 필요 (24시간 주기는 pi/24)
        hour_cos = np.cos(hour * np.pi / 12) * np.float32(np.ones((1, x, y)))
        doy_sin = np.sin(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))
        doy_cos = np.cos(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))

        return np.concatenate([hour_sin, hour_cos, doy_sin, doy_cos]) # 4개 채널 반환

    def __getitem__(self, index):
        """주어진 인덱스에 해당하는 학습 샘플을 반환합니다."""

        # 랜덤 리드 타임 오프셋 설정 (옵션)
        lt_offset = 0
        if self.random_lt:
            # lt_offset 값의 의미 확인 필요 (0, 2, 3이 각각 어떤 시간 오프셋을 나타내는지)
            # 현재는 0, 6시간, 12시간 후 차분을 의미하는 것으로 보임 (frequency=6H 가정)
            lt_offset = np.random.choice([0, 2, 3])

        # 목표 실제값 데이터 로드 (리드 타임 및 오프셋 적용)
        y_target_raw = self.load_era5_time(index + self.lead_time - lt_offset)
        y_target = self.to_tensor(y_target_raw)


        # 초기 조건(컨텍스트) 로드 (ic_path 또는 ERA5 자체)
        if hasattr(self, 'ic') and self.ic is not None: # self.ic_path 대신 self.ic 존재 여부 확인
            # self.ic의 인덱싱 방식 확인 필요 (finetune_step 고려)
            # 현재 코드는 finetune_step과 무관하게 index로 접근
            era5_ts0_raw = self.ic[index].copy().transpose(2, 1, 0) # 축 변경 (채널, 높이, 너비) -> (너비, 높이, 채널) 예상
        else:
            era5_ts0_raw = self.load_era5_time(index)

        # 보조 시간 채널 생성
        time_channels = self.make_time_channels(index, era5_ts0_raw.shape[1], era5_ts0_raw.shape[2]) # era5_ts0_raw 형태에 맞게
        # 컨텍스트 데이터 구성 (초기 조건, 지형, 시간 정보 결합)
        # self.era5_elev의 축 순서 확인 필요 (era5_ts0_raw와 일치해야 함)
        y_context_raw = np.concatenate([era5_ts0_raw, self.era5_elev, time_channels], axis=0) # 채널 축으로 결합
        y_context = self.to_tensor(y_context_raw).permute(0, 2, 1) # (채널, 너비, 높이) -> (채널, 높이, 너비) 예상, 확인 필요

        # 정규화
        if self.diff: # 차분(tendency) 예측
            # 목표값: (미래값 - 현재값) / std_diff
            y_target_tendency = (y_target - self.to_tensor(era5_ts0_raw[:24,...])).permute(2,1,0) # y_target과 era5_ts0_raw의 채널 수 일치 필요
            y_target = self.norm_era5_tendency(y_target_tendency, lt_offset)
            # 컨텍스트: 현재값 / std
            y_context_main_vars = self.norm_era5(y_context[:24, ...]) # 주요 변수만 정규화
            y_context = torch.cat((y_context_main_vars, y_context[24:, ...]), dim=0) # 정규화된 주요 변수와 나머지 보조 변수 결합
        else: # 직접 예측
            if self.norm: # 정규화 사용하는 경우
                y_context_main_vars = self.norm_era5(y_context[:24, ...], lt_offset) # lt_offset 전달
                y_context = torch.cat((y_context_main_vars, y_context[24:, ...]), dim=0)
                y_target = self.norm_era5(y_target, lt_offset) # lt_offset 전달
            y_target = y_target.permute(2, 1, 0) # 축 변경

        if self.rollout: # 전체 타임 시리즈 반환 (옵션)
            targets = []
            # self.lead_time은 6H 단위일 수 있음. range 스텝 확인 필요
            for t_step in range(self.lead_time + 1):
                t_data = self.to_tensor(self.load_era5_time(index + t_step))
                targets.append(t_data.permute(2, 1, 0))
            targets = torch.stack(targets, dim=-1)
            # [..., ::4]는 24시간 간격 샘플링으로 보임 (frequency=6H 가정)
            # frequency=24H일 경우 ::1 이어야 함
            targets_sampled = targets[..., ::(4 if self.frequency == 6 else 1)]


            return {
                "y_context": y_context.permute(0, 2, 1), # 최종 축 순서 확인
                "y_target": y_target,
                "targets": targets_sampled,
                "lt": self.to_tensor(np.array([lt_offset])),
            }
        else:
            return {
                "y_context": y_context.permute(0, 2, 1),
                "y_target": y_target[..., :], # 전체 채널 사용
                "lt": self.to_tensor(np.array([lt_offset])),
                "target_index": self.to_tensor(np.array([index])), # 디버깅 또는 로깅용 인덱스
            }


class WeatherDatasetE2E(WeatherDataset):
    """
    Aardvark 모델의 종단 간(End-to-End) 실행을 위한 데이터셋.
    WeatherDataset을 상속받고, 내부적으로 WeatherDatasetAssimilation, ForecastLoader,
    ForecasterDatasetDownscaling 데이터셋을 사용하여 각 모듈에 필요한 데이터를 제공합니다.
    """

    def __init__(
        self,
        device,
        hadisd_mode, # HadISD 모드 (주로 'train')
        start_date,
        end_date,
        lead_time, # 예측 리드 타임 (일 단위)
        mode, # 데이터 모드 (train, val, test)
        hadisd_var, # 사용할 HadISD 변수
        max_steps_per_epoch=None, # 에포크당 최대 스텝 수 (None이면 전체 데이터 사용)
        era5_mode="sfc",
        res=1,
        filter_dates=None,
        var_start=0, # ERA5 변수 시작 인덱스 (WeatherDatasetAssimilation용)
        var_end=24, # ERA5 변수 끝 인덱스 (WeatherDatasetAssimilation용)
        diff=False, # 차분 사용 여부 (WeatherDatasetAssimilation용)
        two_frames=False, # 두 프레임 사용 여부 (WeatherDatasetAssimilation용)
        region="global", # 지역 (ForecasterDatasetDownscaling용)
    ):

        super().__init__( # WeatherDataset의 __init__ 호출 (기본 설정 로드)
            device,
            hadisd_mode,
            start_date,
            end_date,
            lead_time, # 이 lead_time은 WeatherDataset 내부에서는 사용되지 않을 수 있음
            era5_mode,
            res=res,
            filter_dates=filter_dates,
            diff=diff, # 이 diff도 WeatherDataset 내부에서는 사용되지 않을 수 있음
        )

        # 설정 변수 초기화
        self.var_start = var_start
        self.var_end = var_end
        # self.diff = diff # 이미 부모 클래스에서 설정됨
        self.two_frames = two_frames
        self.region = region
        self.lead_time = lead_time # 종단 간 모델의 최종 리드 타임 (일 단위)
        self.mode = mode
        self.max_steps_per_epoch = max_steps_per_epoch

        # 인코더(동화) 데이터셋 초기화
        self.assimilation_dataset = WeatherDatasetAssimilation(
            device="cuda", # 장치 고정 (설정 가능하게 변경하는 것이 좋음)
            hadisd_mode="train", # 학습 모드로 고정
            start_date=start_date,
            end_date=end_date,
            lead_time=0, # 인코더는 현재 시간 추정이므로 리드 타임 0
            era5_mode="4u", # 4u 모드 사용
            res=1, # 해상도 1도
            var_start=var_start, # 상위 클래스에서 전달받은 값 사용
            var_end=var_end,
            diff=diff, # 상위 클래스에서 전달받은 값 사용
            two_frames=two_frames, # 상위 클래스에서 전달받은 값 사용
        )

        # 예측(프로세서) 데이터셋 초기화
        self.forecast_dataset = ForecastLoader(
            device="cuda", # 장치 고정
            mode=mode, # 상위 클래스에서 전달받은 모드 사용
            lead_time=lead_time, # 최종 리드 타임 (일 단위)
            era5_mode=era5_mode, # 상위 클래스에서 전달받은 ERA5 모드 사용
            res=1, # 해상도 1도
            frequency=6, # 6시간 빈도 고정
            diff=True, # 차분 예측 사용 고정
            u_only=False, # 사용되지 않음
            random_lt=False, # 랜덤 리드 타임 사용 안 함 고정
            # ic_path는 여기서 설정하지 않고, e2e_train.py 등에서 모델 실행 시 주입될 것으로 예상
        )

        # 다운스케일링(디코더) 데이터셋 초기화
        self.downscaling_dataset = ForecasterDatasetDownscaling(
            start_date=start_date,
            end_date=end_date,
            lead_time=lead_time, # 최종 리드 타임 (일 단위)
            hadisd_var=hadisd_var, # 사용할 HadISD 변수
            mode=mode, # 상위 클래스에서 전달받은 모드 사용
            device=device, # 상위 클래스에서 전달받은 장치 사용
            forecast_path=None, # 사전 저장된 예측 경로는 여기서 사용 안 함 (실시간 예측 가정)
            region=region, # 지역 설정
        )

    def __len__(self):
        """데이터셋의 전체 샘플 수를 반환합니다."""
        if self.max_steps_per_epoch: # 최대 스텝 수 제한이 있는 경우
            return self.max_steps_per_epoch
        # 다운스케일링 데이터셋 길이를 기준으로 하되, 10일 오프셋 적용
        return len(self.downscaling_dataset) - 40

    def __getitem__(self, index):
        """주어진 인덱스에 해당하는 종단 간 학습 샘플을 반환합니다."""

        if self.max_steps_per_epoch: # 최대 스텝 수 제한 시 랜덤 샘플링
            index = np.random.choice(
                np.arange(len(self.downscaling_dataset) - 40) # 오프셋 적용된 범위 내에서 랜덤 선택
            )

        # 각 모듈별 데이터 가져오기
        # assimilation_dataset의 인덱스와 forecast/downscaling 인덱스가 동일하게 사용됨
        # 이는 각 모듈이 동일한 초기 시간에서 시작하여 순차적으로 예측한다고 가정
        assimilation_data = self.assimilation_dataset.__getitem__(index)
        forecast_data = self.forecast_dataset.__getitem__(index) # ForecastLoader의 lead_time은 일 단위
        downscaling_data = self.downscaling_dataset.__getitem__(index) # lead_time은 일 단위

        # 모든 데이터를 하나의 딕셔너리로 묶음
        task = {
            "assimilation": assimilation_data, # 인코더 입력/목표
            "forecast": forecast_data, # 프로세서 입력/목표
            "downscaling": downscaling_data, # 디코더 입력/목표
            "index": torch.tensor(index), # 현재 인덱스 (디버깅/로깅용)
        }

        # 종단 간 미세조정을 위해 최종 목표값(다운스케일링 결과)을 최상위 키로 추가
        task["y_target"] = task["downscaling"]["y_target"]

        return task

    def unnorm_pred(self, x):
        """
        정규화된 (최종) 예측값을 원래 스케일로 복원합니다.
        WeatherDatasetAssimilation의 unnorm_pred를 사용하며, 이는 ERA5 기준의 역정규화입니다.
        다운스케일링 결과가 HadISD 기준이라면, downscaling_dataset.unnorm_pred를 사용해야 할 수 있습니다.
        현재 코드는 WeatherDataset (부모 클래스)의 unnorm_pred를 호출하게 되어 ERA5 기준.
        """
        # 부모 클래스 WeatherDataset의 unnorm_era5가 아닌,
        # WeatherDatasetAssimilation에서 오버라이드된 unnorm_pred를 사용해야 함.
        # 하지만, 이 클래스는 WeatherDataset을 직접 상속하므로, WeatherDataset의 unnorm_era5를 사용하게 됨.
        # 이는 self.means, self.stds, self.diff, self.era5_mean_spatial 등을 WeatherDataset의 것을 사용.
        # 의도된 동작인지 확인 필요. 만약 assimilation_dataset의 역정규화를 원한다면 해당 객체 메서드 호출 필요.

        dev = x.device
        x = x.detach().cpu().numpy()

        # WeatherDataset의 정규화 계수(self.means, self.stds) 사용
        x = (
            x
            * self.stds[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end # var_start, var_end는 assimilation용
            ]
            + self.means[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
        )
        if bool(self.diff): # self.diff도 assimilation용
            x = (
                x
                + self.era5_mean_spatial[np.newaxis, ...].transpose(0, 3, 2, 1)[
                    ..., self.var_start : self.var_end
                ]
            )
        return torch.from_numpy(x).float().to(dev)
