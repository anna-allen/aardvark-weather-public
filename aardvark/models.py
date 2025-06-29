# 시스템 경로 관련 모듈 (여기서는 상위 디렉토리 추가에 사용)
import sys

# NumPy, PyTorch 라이브러리 임포트
import numpy as np
import torch
import torch.nn as nn

# 현재 디렉토리 및 하위 디렉토리의 모듈 임포트
from architectures import MLP  # MLP 아키텍처
from set_convs import convDeepSet  # Set convolution 연산
from unet_wrap_padding import *  # U-Net 관련 패딩 래퍼 (내용 확인 필요)
from vit import *  # Vision Transformer 관련 모듈 (내용 확인 필요)

# 상위 디렉토리를 시스템 경로에 추가 (다른 모듈 임포트를 위함일 수 있음)
sys.path.append("../")


class ConvCNPWeather(nn.Module):
    """
    ConvCNP (Convolutional Conditional Neural Process) 기반의 날씨 모델 클래스.
    주로 인코더 및 프로세서 모듈에 사용됩니다.
    다양한 기상 관측 데이터를 입력으로 받아 처리하고,
    Vision Transformer (ViT)와 같은 백본 네트워크를 사용하여 예측을 생성합니다.
    """

    def __init__(
        self,
        in_channels,  # 입력 채널 수
        out_channels,  # 출력 채널 수
        int_channels,  # 내부 중간 채널 수
        device,  # 사용할 장치 (e.g., "cuda", "cpu")
        res,  # 해상도 (데이터 로딩 시 사용)
        data_path="../data/",  # 데이터 기본 경로
        gnp=False,  # Graph Neural Process 사용 여부 (여기서는 직접 사용되지 않는 듯)
        mode="assimilation",  # 모델 운영 모드 ("assimilation" 또는 "forecast")
        decoder=None,  # 사용할 디코더 타입 (ConvCNP의 "디코더", 즉 백본 네트워크)
        film=False,  # FiLM (Feature-wise Linear Modulation) 레이어 사용 여부
        two_frames=False,  # 두 개의 시간 프레임(t-1, t)을 입력으로 사용할지 여부
    ):

        super().__init__()  # 부모 클래스 초기화

        # 클래스 멤버 변수 설정
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.decoder = decoder  # 여기서 'decoder'는 ConvCNP 문맥에서의 디코더(ViT 등)
        self.int_x = 256  # 내부 그리드 x 크기 (고정값)
        self.int_y = 128  # 내부 그리드 y 크기 (고정값)
        self.data_path = data_path
        self.mode = mode
        self.film = film
        self.two_frames = two_frames

        # 각 데이터 소스별 변수 수 정의 (상수)
        N_SAT_VARS = 2  # 위성(GRIDSAT) 변수 수
        N_ICOADS_VARS = 5  # ICOADS 변수 수
        N_HADISD_VARS = 5  # HadISD 변수 수 (loader.py와 일치해야 함)

        # 내부 그리드의 경도/위도 위치 정보 로드 및 정규화
        self.era5_x = (
            torch.from_numpy(
                np.load(self.data_path + "grid_lon_lat/era5_x_{}.npy".format(res))
            ).float()
            / 360  # LATLON_SCALE_FACTOR (360)로 나누어 [0,1] 범위로 스케일링 가정
        )
        self.era5_y = (
            torch.from_numpy(
                np.load(self.data_path + "grid_lon_lat/era5_y_{}.npy".format(res))
            ).float()
            / 360
        )

        # 모델 내부에서 사용할 목표 그리드 정의 (정규화된 좌표)
        self.int_grid = [
            (torch.linspace(0, 360, 240) / 360).float().to(self.device), # 경도, to(self.device)로 수정
            (torch.linspace(-90, 90, 121) / 360).float().to(self.device), # 위도, to(self.device)로 수정
        ]
        # 배치 차원 추가
        self.int_grid = [self.int_grid[0].unsqueeze(0), self.int_grid[1].unsqueeze(0)]

        # 각 데이터 소스별 입력 처리를 위한 Set Convolution 레이어 생성
        # convDeepSet: 불규칙한 관측 지점 데이터를 규칙적인 그리드 표현으로 변환
        # density_channel=True: 밀도 채널 사용 여부
        # "OnToOn": 그리드에서 그리드로 (주로 규칙적 데이터)
        # "OffToOn": 관측 지점에서 그리드로 (주로 불규칙적 데이터)
        self.ascat_setconvs = convDeepSet(
            0.001, "OnToOn", density_channel=True, device=self.device
        )
        self.amsua_setconvs = nn.ModuleList([ # nn.ModuleList로 감싸서 PyTorch가 인식하도록 함
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(13)  # AMSU-A 채널 수만큼 생성
        ])
        self.amsub_setconvs = nn.ModuleList([
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(12)  # AMSU-B 채널 수만큼 생성
        ])
        self.hirs_setconvs = nn.ModuleList([
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(26)  # HIRS 채널 수만큼 생성
        ])
        self.sat_setconvs = nn.ModuleList([
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(N_SAT_VARS)
        ])
        self.hadisd_setconvs = nn.ModuleList([
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_HADISD_VARS)
        ])
        self.icoads_setconvs = nn.ModuleList([
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_ICOADS_VARS)
        ])
        self.igra_setconvs = nn.ModuleList([
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(24)  # IGRA 채널 수만큼 생성
        ])

        # 출력 Set Convolution 레이어 (여기서는 직접 사용되지 않는 듯)
        self.sc_out = convDeepSet(
            0.001, "OnToOff", density_channel=False, device=self.device
        )

        # ConvCNP의 디코더(백본 네트워크) 인스턴스화
        if self.decoder == "vit":
            # Vision Transformer (ViT) 사용
            self.decoder_lr = ViT(
                in_channels=in_channels,  # 입력 채널 수는 다양한 인코딩 결합 후 결정됨
                out_channels=out_channels,
                h_channels=512,  # ViT 내부 히든 채널 수
                depth=16,  # ViT 트랜스포머 블록 깊이
                patch_size=5,
                per_var_embedding=True,  # 변수별 임베딩 사용 여부
                img_size=[240, 121],  # 입력 이미지 크기
            )
        elif self.decoder == "vit_assimilation":
            # 동화(assimilation) 모드용 ViT (더 작은 설정)
            self.decoder_lr = ViT(
                in_channels=256,  # 고정된 입력 채널 수
                out_channels=out_channels,
                h_channels=512,
                depth=8,
                patch_size=3,
                per_var_embedding=False,
                img_size=[256, 128],  # 내부 그리드 크기와 일치
            )
        # else: self.decoder_lr가 정의되지 않을 수 있음. 오류 처리 필요.

        # MLP는 현재 코드에서 직접 사용되지 않음 (초기화는 되어 있음)
        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            h_channels=128,
            h_layers=4,
        )
        self.break_next = False # 디버깅용 플래그로 보임

    def encoder_hadisd(self, task, prefix):
        """HadISD 데이터에 대한 전처리 및 인코딩을 수행합니다."""
        encodings = []
        # HadISD는 5개 변수, 여기서는 4개 채널만 처리 (0~3). loader.py의 N_HADISD_VARS와 비교 필요.
        for channel in range(4): # 0, 1, 2, 3 채널만 사용
            encodings.append(
                self.hadisd_setconvs[channel]( # 각 채널별 SetConv 사용
                    x_in=[ # 입력 위치 (경도, 위도)
                        task[f"x_context_hadisd_{prefix}"][channel][:, 0, :],
                        task[f"x_context_hadisd_{prefix}"][channel][:, 1, :],
                    ],
                    wt=task[f"y_context_hadisd_{prefix}"][channel].unsqueeze(1), # 입력 관측값
                    x_out=self.int_grid,  # 목표 그리드
                )
            )
        encodings = torch.cat(encodings, dim=1)  # 채널 차원으로 결합
        return encodings

    def encoder_sat(self, task, prefix):
        """GRIDSAT (위성) 데이터에 대한 전처리 및 인코딩을 수행합니다."""
        encodings = []
        for channel in range(task[f"sat_{prefix}"].shape[1]): # 모든 채널에 대해 반복
            encodings.append(
                self.sat_setconvs[channel](
                    x_in=task[f"sat_x_{prefix}"],
                    wt=task[f"sat_{prefix}"][:, channel : channel + 1, ...], # 각 채널 데이터
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_icoads(self, task, prefix):
        """ICOADS 데이터에 대한 전처리 및 인코딩을 수행합니다."""
        encodings = []
        for channel in range(N_ICOADS_VARS): # 정의된 변수 수만큼 반복
            encodings.append(
                self.icoads_setconvs[channel](
                    x_in=task[f"icoads_x_{prefix}"],
                    wt=task[f"icoads_{prefix}"][:, channel, :].unsqueeze(1), # 관측소 차원 처리
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_amsua(self, task, prefix):
        """AMSU-A 데이터에 대한 전처리 및 인코딩을 수행합니다."""
        encodings = []
        # 결측값 처리 (NaN으로 변경)
        task[f"amsua_{prefix}"][..., -1] = np.nan # 마지막 채널을 NaN으로 (이유 확인 필요)
        task[f"amsua_{prefix}"][task[f"amsua_{prefix}"] == 0] = np.nan # 0 값을 NaN으로
        for i in range(13): # AMSU-A 채널 수 (13)
            encodings.append(
                self.amsua_setconvs[i](
                    x_in=task[f"amsua_x_{prefix}"],
                    # 축 순서 변경 (배치, 채널, 위도, 경도) 형태로 맞춤
                    wt=task[f"amsua_{prefix}"].permute(0, 3, 2, 1)[:, i : i + 1, ...],
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_amsub(self, task, prefix):
        """AMSU-B 데이터에 대한 전처리 및 인코딩을 수행합니다."""
        encodings = []
        task[f"amsub_{prefix}"][task[f"amsub_{prefix}"] == 0] = np.nan # 0 값을 NaN으로
        for i in range(12): # AMSU-B 채널 수 (12)
            # AMSU-A SetConv 재활용 (amsua_setconvs[i]). AMSU-B용 SetConv가 별도로 있어야 할 수 있음.
            encodings.append(
                self.amsua_setconvs[i]( # AMSU-B용 setconvs 사용해야 함: self.amsub_setconvs[i]
                    x_in=task[f"amsub_x_{prefix}"],
                    # 축 순서 변경
                    wt=task[f"amsub_{prefix}"].permute(0, 3, 1, 2)[:, i : i + 1, ...],
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_hirs(self, task, prefix):
        """HIRS 데이터에 대한 전처리 및 인코딩을 수행합니다."""
        encodings = []
        task[f"hirs_{prefix}"][task[f"hirs_{prefix}"] == 0] = np.nan # 0 값을 NaN으로
        for i in range(26): # HIRS 채널 수 (26)
            encodings.append(
                self.hirs_setconvs[i](
                    x_in=task[f"hirs_x_{prefix}"],
                    # 축 순서 변경
                    wt=task[f"hirs_{prefix}"].permute(0, 3, 1, 2)[:, i : i + 1, ...],
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_igra(self, task, prefix):
        """IGRA 데이터에 대한 전처리 및 인코딩을 수행합니다."""
        encodings = []
        for channel in range(24): # IGRA 채널 수 (24)
            encodings.append(
                self.igra_setconvs[channel](
                    x_in=task[f"igra_x_{prefix}"],
                    wt=task[f"igra_{prefix}"][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_ascat(self, task, prefix):
        """ASCAT 데이터에 대한 전처리. SetConv 대신 보간법 사용."""
        # NaN 값을 0으로 채우고 보간 수행. 이후 그리드 뒤집기.
        # SetConv를 사용하지 않는 이유 확인 필요 (데이터 특성 또는 다른 처리 방식).
        task[f"ascat_{prefix}"][torch.isnan(task[f"ascat_{prefix}"])] = 0
        # 축 순서 변경 후 보간 (240x121 크기로)
        e = nn.functional.interpolate(
            task[f"ascat_{prefix}"].permute(0, 3, 1, 2), size=(240, 121)
        )
        e = torch.flip(e, dims=[-1]) # 마지막 차원(위도) 뒤집기
        return e

    def encoder_iasi(self, task, prefix):
        """IASI 데이터에 대한 전처리. SetConv 대신 보간법 사용."""
        task[f"iasi_{prefix}"][torch.isnan(task[f"iasi_{prefix}"])] = 0
        e = nn.functional.interpolate(
            task[f"iasi_{prefix}"].permute(0, 3, 1, 2), size=(240, 121)
        )
        e = torch.flip(e, dims=[-1])
        return e

    def forward(self, task, film_index):
        """모델의 순전파 로직을 정의합니다."""

        # 입력 데이터 설정
        if self.mode == "assimilation":  # 동화 모드일 경우
            # 내부 그리드를 현재 타겟 장치로 이동
            self.int_grid = [grid_dim.to(task["y_target_current"].device) for grid_dim in self.int_grid] # y_target -> y_target_current

            # 지형(고도) 데이터 처리: 보간 및 축 변경
            # task["era5_elev_current"] 사용
            elev = nn.functional.interpolate(
                torch.flip(task["era5_elev_current"].permute(0, 1, 3, 2), dims=[2]), # 축 변경 후 뒤집기
                size=(self.int_grid[1].shape[1], self.int_grid[0].shape[1]), # (위도, 경도) 크기로 보간 (순서 확인)
            ) # 원본 코드에서는 int_grid[0] (경도), int_grid[1] (위도) 순서였음. interpolate size는 (H, W)

            if not self.two_frames: # 단일 시간 프레임 입력
                # 각 데이터 소스별 인코더 함수 호출하여 인코딩된 특징 생성
                encodings = [
                    self.encoder_iasi(task, "current"),
                    self.encoder_ascat(task, "current"),
                    self.encoder_hadisd(task, "current"),
                    self.encoder_icoads(task, "current"),
                    self.encoder_sat(task, "current"),
                    self.encoder_amsua(task, "current"),
                    self.encoder_amsub(task, "current"), # AMSU-B 인코더 호출 확인 (현재 amsua_setconvs 사용 중)
                    self.encoder_igra(task, "current"),
                    self.encoder_hirs(task, "current"),
                    elev,  # 처리된 지형 데이터
                    task["climatology_current"],  # 현재 시간의 기후학 데이터
                    # 시간 보조 채널 (5개 채널로 확장하여 elev와 유사한 형태)
                    torch.ones_like(elev[:, :5, ...]) # elev의 처음 5개 채널과 같은 형태로 만듦
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
            else:  # 두 개의 시간 프레임(t-1, t) 입력
                encodings = [
                    # 현재(current) 시간 데이터 인코딩
                    self.encoder_iasi(task, "current"),
                    self.encoder_ascat(task, "current"),
                    self.encoder_hadisd(task, "current"),
                    self.encoder_icoads(task, "current"),
                    self.encoder_sat(task, "current"),
                    self.encoder_amsua(task, "current"),
                    self.encoder_amsub(task, "current"),
                    self.encoder_igra(task, "current"),
                    self.encoder_hirs(task, "current"),
                    # 이전(prev) 시간 데이터 인코딩
                    self.encoder_iasi(task, "prev"),
                    self.encoder_ascat(task, "prev"),
                    self.encoder_hadisd(task, "prev"),
                    self.encoder_icoads(task, "prev"),
                    self.encoder_sat(task, "prev"),
                    self.encoder_amsua(task, "prev"),
                    self.encoder_amsub(task, "prev"),
                    self.encoder_igra(task, "prev"),
                    self.encoder_hirs(task, "prev"),
                    elev,
                    task["climatology_current"],
                    torch.ones_like(elev[:, :5, ...])
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
            x = torch.cat(encodings, dim=1)  # 모든 인코딩된 특징을 채널 차원으로 결합

        else:  # 예보(forecast) 모드일 경우
            x = task["y_context"]  # 컨텍스트 데이터를 입력으로 사용

        # 입력 데이터의 축 순서 확인 및 조정 (채널, 높이, 너비) 형태로
        if x.shape[-1] > x.shape[-2]: # 너비가 높이보다 크면 (경도 > 위도)
            x = x.permute(0, 1, 3, 2) # (배치, 채널, 위도, 경도) -> (배치, 채널, 경도, 위도) - 확인 필요. ViT 입력은 (B, C, H, W)

        # ViT 백본 네트워크 실행
        if self.decoder == "vit": # 일반 ViT
            # lead_times는 ViT 내부에서 시간 정보를 활용하는 데 사용될 수 있음
            x = self.decoder_lr(x, lead_times=task["lt"])
            x = x.permute(0, 3, 1, 2) # ViT 출력 (B, H, W, C) -> (B, C, H, W) 형태로 변경 가정
        elif self.decoder == "vit_assimilation": # 동화 모드용 ViT
            # 입력 크기를 (256, 128)로 보간
            x = nn.functional.interpolate(x, size=(self.int_y, self.int_x)) # (H, W) 순서
            # film_index는 FiLM 레이어 사용 시 전달 (여기서는 (lt * 0) + 1 로 고정된 값 사용)
            x = self.decoder_lr(x, film_index=(task["lt"] * 0) + 1)
        # else: self.decoder_lr가 정의되지 않은 경우 처리 누락

        # 출력 처리
        if np.logical_and(
            self.mode == "assimilation", self.decoder == "vit_assimilation"
        ):
            # 동화 모드 및 vit_assimilation 디코더 사용 시
            # 출력을 (240, 121) 크기로 보간하고 축 순서 변경
            x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(240, 121)) # (B,C,H,W) 가정
            return x.permute(0, 3, 2, 1) # (B, W, H, C) 형태로 반환 가정

        elif self.mode == "forecast":
            # 예보 모드 시
            # 출력을 (240, 121) 크기로 보간하고 축 순서 변경
            x = nn.functional.interpolate(x, size=(121, 240)).permute(0, 2, 3, 1) # (B, H, W, C)
            return x.permute(0, 2, 1, 3) # (B, W, H, C) 형태로 반환 가정

        # 그 외의 경우 (예: self.mode == "assimilation" and self.decoder == "vit")
        return x # 현재 x의 형태는 (B, C, H, W) 또는 ViT의 출력 형태에 따라 다름
