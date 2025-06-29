# math 라이브러리 임포트 (ceil 함수 사용)
import math

# PyTorch 라이브러리 임포트
import torch
import torch.nn as nn


def cylindrical_conv_pad(x, w_pad):
    """
    원통형(cylindrical) 경계 조건에 대한 컨볼루션 연산을 위해 너비(경도) 방향으로 패딩을 적용합니다.
    데이터의 왼쪽 끝과 오른쪽 끝이 연결되어 있다고 가정하고, 양쪽에서 데이터를 가져와 이어 붙입니다.

    Args:
        x (torch.Tensor): 입력 텐서 (배치, 채널, 높이, 너비).
        w_pad (int): 너비 방향으로 패딩할 크기.

    Returns:
        torch.Tensor: 너비 방향으로 원통형 패딩이 적용된 텐서.
    """
    # x[..., -w_pad:] : 너비의 마지막 w_pad 만큼을 가져옴 (오른쪽 끝)
    # x[..., :w_pad]  : 너비의 처음 w_pad 만큼을 가져옴 (왼쪽 끝)
    # [오른쪽 끝, 원본, 왼쪽 끝] 순서로 너비(axis=-1)를 따라 이어 붙임.
    return torch.cat([x[..., -w_pad:], x, x[..., :w_pad]], axis=-1)


class CylindricalConv2D(nn.Conv2d):
    """
    원통형 경계 조건을 사용하는 2D 컨볼루션 레이어.
    `nn.Conv2d`를 상속받아 `forward` 메소드를 오버라이드하여 원통형 패딩을 적용합니다.
    주로 U-Net과 같은 아키텍처에서 경도 방향의 주기성을 처리하기 위해 사용됩니다.
    """

    def __init__(
        self,
        in_channels: int,  # 입력 채널 수
        out_channels: int,  # 출력 채널 수
        kernel_size: int,  # 커널 크기 (단일 정수 또는 튜플)
        stride: int,  # 스트라이드 (단일 정수 또는 튜플)
    ):
        # 부모 클래스(nn.Conv2d)의 초기화 함수 호출
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        # 커널 크기는 홀수여야 함 (패딩 계산을 용이하게 하기 위함)
        assert self.kernel_size[0] % 2 == 1, "커널 높이는 홀수여야 합니다."
        assert self.kernel_size[1] % 2 == 1, "커널 너비는 홀수여야 합니다."

        # 높이 및 너비 방향 패딩 크기 계산 (커널 크기의 절반)
        self.h_pad = self.kernel_size[0] // 2
        self.w_pad = self.kernel_size[1] // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 연산을 수행합니다.
        먼저 높이 방향으로 일반적인 제로 패딩을 적용하고,
        그 다음 너비 방향으로 원통형 패딩을 적용한 후,
        부모 클래스의 컨볼루션 연산을 수행합니다.
        """
        # 높이 방향으로 제로 패딩 적용 (위, 아래)
        # (패딩_왼쪽, 패딩_오른쪽, 패딩_위, 패딩_아래) 순서
        x = nn.functional.pad(x, (0, 0, self.h_pad, self.h_pad))
        # 너비 방향으로 원통형 패딩 적용 후 부모 클래스의 forward 호출
        return super().forward(cylindrical_conv_pad(x, self.w_pad))


class CylindricalConvTranspose2D(nn.ConvTranspose2d):
    """
    원통형 경계 조건을 사용하는 2D 전치 컨볼루션(Transposed Convolution) 레이어.
    `nn.ConvTranspose2d`를 상속받아 `forward` 메소드를 오버라이드합니다.
    업샘플링 과정에서 원통형 경계 조건을 유지하기 위해 사용됩니다.
    패딩 및 출력 크롭핑 로직이 포함되어 있습니다.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int, # 단일 정수 또는 튜플
        stride: int, # 단일 정수 또는 튜플
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            # ConvTranspose2d는 output_padding, padding 등을 자동으로 계산하거나
            # 명시적으로 지정해야 원하는 출력 크기를 얻을 수 있음.
            # 여기서는 bias를 직접 관리하고, forward에서 크롭핑을 통해 출력 크기 조절.
            bias=False # 부모 클래스의 bias는 사용하지 않고 아래에서 _bias를 직접 정의
        )

        # 커널 크기는 홀수여야 함
        assert self.kernel_size[0] % 2 == 1, "커널 높이는 홀수여야 합니다."
        assert self.kernel_size[1] % 2 == 1, "커널 너비는 홀수여야 합니다."

        # 스트라이드 및 커널 크기 저장
        self.sh, self.sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.kh, self.kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)

        # 필요한 입력 패딩 크기 계산 (원통형 패딩 전의 입력에 대한 패딩)
        # 이 계산은 전치 컨볼루션 후 원하는 출력 크기를 얻기 위함으로 보임.
        # 일반적인 전치 컨볼루션의 출력 크기 공식과 관련하여 패딩을 역산하는 과정일 수 있음.
        # H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
        # 여기서 self.h_pad는 입력에 적용할 패딩을 의미하는 것으로 보임.
        # 패딩 계산이 복잡하며, 특정 출력 크기를 맞추기 위한 것일 수 있음.
        self.h_pad = math.ceil(((self.sh - 1) + 2 * (self.kh // 2)) / self.sh)
        self.w_pad = math.ceil(((self.sw - 1) + 2 * (self.kw // 2)) / self.sw)

        # 컨볼루션 후 크롭핑을 위한 시작 인덱스 계산
        self.h0 = self.sh * self.h_pad - (self.sh - 1) + (self.kh // 2)
        self.w0 = self.sw * self.w_pad - (self.sw - 1) + (self.kw // 2)

        # 커스텀 바이어스 파라미터 (작은 값으로 초기화)
        self._bias = nn.Parameter(10**-3 * torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 연산을 수행합니다.
        입력에 원통형 패딩 및 일반 패딩을 적용하고,
        부모 클래스의 전치 컨볼루션 연산을 수행한 후,
        계산된 오프셋을 사용하여 원하는 출력 크기로 크롭핑하고 바이어스를 더합니다.
        """
        # 목표 출력 높이 및 너비 계산
        Nh = x.shape[2] * self.sh
        Nw = x.shape[3] * self.sw

        # 너비 방향 원통형 패딩
        x = cylindrical_conv_pad(x, self.w_pad)
        # 높이 방향 제로 패딩
        x = nn.functional.pad(x, (0, 0, self.h_pad, self.h_pad))

        # 부모 클래스의 전치 컨볼루션 연산 수행
        x = super().forward(x)

        # 계산된 오프셋(h0, w0)과 목표 크기(Nh, Nw)를 사용하여 출력 크롭핑
        # 전치 컨볼루션 결과에서 유효한 부분만 선택
        return x[:, :, self.h0 : self.h0 + Nh, self.w0 : self.w0 + Nw] + self._bias.view(1, -1, 1, 1)


class Down(nn.Module):
    """
    U-Net의 다운샘플링(인코더) 경로를 구성하는 한 블록입니다.
    두 개의 CylindricalConv2D 레이어, 배치 정규화, 활성화 함수(GELU)를 포함합니다.
    선택적으로 FiLM(Feature-wise Linear Modulation) 레이어와 어텐션 블록을 포함할 수 있습니다.
    두 번째 컨볼루션 레이어는 스트라이드를 통해 다운샘플링을 수행할 수 있습니다.
    """

    def __init__(
        self,
        in_channels,  # 입력 채널 수
        out_channels,  # 출력 채널 수
        p=0,  # 사용되지 않는 파라미터 (드롭아웃 확률이었을 수 있음)
        film=False,  # FiLM 레이어 사용 여부
        down=True,  # 두 번째 컨볼루션에서 다운샘플링 수행 여부
        attn=False,  # 어텐션 블록 사용 여부 (AttentionBlock 클래스 필요)
    ):

        super().__init__()

        self.film = film
        self.attn = attn # 어텐션 사용 여부 저장

        # 첫 번째 원통형 컨볼루션 (스트라이드 1)
        self.conv_1 = CylindricalConv2D(
            in_channels, out_channels, kernel_size=3, stride=1
        )
        # 두 번째 원통형 컨볼루션 (다운샘플링 여부에 따라 스트라이드 결정)
        if down:
            self.conv_2 = CylindricalConv2D(
                out_channels, out_channels, kernel_size=3, stride=2  # 다운샘플링
            )
        else:
            self.conv_2 = CylindricalConv2D(
                out_channels, out_channels, kernel_size=3, stride=1  # 다운샘플링 안 함
            )

        # 배치 정규화 레이어
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        # 활성화 함수 (GELU)
        self.activation = nn.GELU()

        # FiLM 레이어 사용 시 필요한 파라미터 (gamma, beta)
        if film:
            # film_index (최대 10개로 가정)에 따른 gamma, beta 값들
            self.gamma_1 = torch.nn.Parameter(
                torch.ones(10, out_channels, 1, 1), requires_grad=True # requires_grad 명시
            )
            self.gamma_2 = torch.nn.Parameter(
                torch.ones(10, out_channels, 1, 1), requires_grad=True
            )
            self.beta_1 = torch.nn.Parameter(
                torch.zeros(10, out_channels, 1, 1), requires_grad=True
            )
            self.beta_2 = torch.nn.Parameter(
                torch.zeros(10, out_channels, 1, 1), requires_grad=True
            )

        # 어텐션 블록 사용 시 (AttentionBlock 클래스가 정의되어 있어야 함)
        if self.attn:
            # from vit import AttentionBlock 과 같이 임포트 필요
            # 여기서는 AttentionBlock이 현재 파일에 없으므로 주석 처리 또는 임포트 필요
            # self.mha = AttentionBlock(n_channels=out_channels, n_heads=8)
            pass # 임시로 pass 처리

    def forward(self, xi, film_index=None):
        """순전파 연산을 정의합니다."""

        # FiLM 사용 시 film_index에서 실제 인덱스 추출
        if self.film and film_index is None:
            raise ValueError("FiLM을 사용하는 경우 film_index를 제공해야 합니다.")
        if self.film:
            # film_index가 텐서일 경우를 대비하여 .long() 또는 .int()로 변환
            film_idx = film_index[:, 0].long() # film_index가 (배치, 1) 형태일 것으로 가정

        # 첫 번째 컨볼루션 블록
        x = self.conv_1(xi)
        x = self.bn_1(x)
        if self.film:
            # 해당 film_index에 맞는 gamma, beta 선택
            g1 = torch.index_select(self.gamma_1, 0, film_idx)
            b1 = torch.index_select(self.beta_1, 0, film_idx)
            x = g1 * x + b1  # FiLM 적용
        x = self.activation(x)

        # 두 번째 컨볼루션 블록
        x = self.conv_2(x)
        x = self.bn_2(x)
        if self.film:
            g2 = torch.index_select(self.gamma_2, 0, film_idx)
            b2 = torch.index_select(self.beta_2, 0, film_idx)
            x = g2 * x + b2  # FiLM 적용
        x = self.activation(x)

        # 어텐션 블록 적용 (선택 사항)
        if self.attn and hasattr(self, 'mha'): # mha가 정의되어 있을 때만 실행
            x = self.mha(x)

        return x


class Up(nn.Module):
    """
    U-Net의 업샘플링(디코더) 경로를 구성하는 한 블록입니다.
    업샘플링 레이어(Bilinear Upsampling 또는 CylindricalConvTranspose2D)와
    Down 클래스(여기서는 다운샘플링 없이 사용)를 사용하여 컨볼루션 연산을 수행합니다.
    U-Net의 skip connection을 위해 두 개의 입력(x1: 업샘플링 대상, x2: skip connection)을 받습니다.
    """

    def __init__(
        self,
        in_channels,  # 업샘플링 레이어의 입력 채널 수 (Down 블록의 출력 채널 수)
        out_channels,  # 최종 출력 채널 수 (Down 블록 내부의 출력 채널 수)
        p,  # 사용되지 않는 파라미터
        bilinear=False,  # Bilinear Upsampling 사용 여부
        film=False,  # FiLM 레이어 사용 여부
        stride=2,  # CylindricalConvTranspose2D 사용 시 스트라이드
        attn=False,  # 어텐션 블록 사용 여부
    ):
        super().__init__()

        self.film = film

        # 업샘플링 레이어 설정
        if bilinear:
            # Bilinear Upsampling 사용 (채널 수 변경 없음)
            self.up = nn.Upsample(
                scale_factor=2,  # 크기를 2배로
                mode="bilinear",
                align_corners=True,  # 코너 픽셀 정렬 방식
            )
            # Bilinear 사용 시 conv의 입력 채널은 in_channels 그대로 사용
            conv_in_channels = in_channels
        else:
            # CylindricalConvTranspose2D 사용 (채널 수를 out_channels로 변경)
            self.up = CylindricalConvTranspose2D(
                in_channels, out_channels, kernel_size=3, stride=stride
            )
            # ConvTranspose2D가 채널 수를 out_channels로 변경하므로 conv의 입력은 out_channels
            conv_in_channels = out_channels


        # 업샘플링 후 컨볼루션 연산을 위한 Down 블록 (다운샘플링 없이 사용)
        self.conv = Down(
            conv_in_channels, # Bilinear일 경우 in_channels, ConvTranspose2D일 경우 out_channels
            out_channels,    # Down 블록의 최종 출력 채널
            p=0, # Down 클래스의 p는 사용되지 않음
            film=film,
            down=False,  # 다운샘플링 안 함
            attn=attn,
        )

    def forward(self, x1, x2, film_index=None):
        """
        순전파 연산을 정의합니다.
        x1: 업샘플링될 낮은 해상도의 특징 맵.
        x2: U-Net의 인코더 경로에서 전달된 높은 해상도의 특징 맵 (skip connection).
        """
        x1 = self.up(x1)  # x1 업샘플링

        # 업샘플링 후 컨볼루션 수행
        # 만약 bilinear=True이고 up이 채널 수를 변경하지 않았다면,
        # self.conv의 입력 채널 수가 x1의 채널 수와 일치해야 함.
        # 현재 self.conv는 in_channels를 입력으로 받도록 되어 있음.
        # bilinear=True일 때, self.conv의 입력 채널은 in_channels.
        # bilinear=False일 때, self.up이 out_channels로 변경하므로 self.conv의 입력은 out_channels.
        # __init__에서 conv_in_channels로 이를 처리함.
        x1 = self.conv(x1, film_index=film_index)

        # 업샘플링된 x1과 skip connection으로 전달된 x2의 크기를 맞추기 위한 크롭핑.
        # 경계 조건 등으로 인해 크기가 1픽셀 정도 차이 날 수 있음.
        if x1.shape[-1] != x2.shape[-1]: # 너비가 다를 경우
            x1 = x1[..., :, :x2.shape[-1]] # x2의 너비에 맞게 x1을 크롭 (오른쪽 끝을 자름)
        if x1.shape[-2] != x2.shape[-2]: # 높이가 다를 경우
            x1 = x1[..., :x2.shape[-2], :] # x2의 높이에 맞게 x1을 크롭 (아래쪽 끝을 자름)

        # x2 (skip connection)와 처리된 x1을 채널 차원으로 결합.
        return torch.cat([x2, x1], dim=1)


class Unet(nn.Module):
    """
    원통형 경계 조건을 사용하는 U-Net 모델 아키텍처.
    Down 및 Up 블록을 사용하여 인코더-디코더 구조를 형성합니다.
    """

    def __init__(
        self,
        in_channels,  # 입력 채널 수
        out_channels,  # 최종 출력 채널 수
        div_factor=1,  # 채널 수 조절 인자 (채널 수를 나눔)
        p=0.0,  # 사용되지 않는 파라미터 (드롭아웃 확률이었을 수 있음)
        context=True,  # 사용되지 않는 파라미터
        film=False,  # FiLM 레이어 사용 여부
        film_base=True,  # 사용되지 않는 파라미터
    ):
        super(Unet, self).__init__()

        self.n_channels = in_channels
        self.bilinear = True # Up 블록에서 Bilinear Upsampling 사용 여부 (여기서는 True로 고정, Up 클래스 내부에서 결정)
        self.fp = nn.Softplus() # Softplus 활성화 함수 (여기서는 직접 사용되지 않음)
        # 출력 분산을 위한 학습 가능한 파라미터 (여기서는 직접 사용되지 않음)
        self.variances = nn.Parameter(torch.zeros([out_channels]))
        self.context = context # 사용되지 않음
        self.film = film

        m = 1  # 채널 수 배율 인자 (현재는 1로 고정)

        # U-Net 인코더 (다운샘플링 경로)
        self.down1 = Down(self.n_channels, m * 128 // div_factor, p=0, film=film, attn=False)
        self.down2 = Down(m * 128 // div_factor, m * 256 // div_factor, p=0, film=film, attn=False)
        self.down3 = Down(m * 256 // div_factor, m * 512 // div_factor, p=0, film=film, attn=False)
        # 가장 깊은 곳의 다운샘플링 블록 (해상도 유지 또는 추가 다운샘플링 가능)
        self.down4 = Down(m * 512 // div_factor, m * 512 // div_factor, p=0, film=film, attn=False)

        # U-Net 디코더 (업샘플링 경로)
        # Up 블록의 in_channels는 이전 Up 블록의 out_channels + skip connection 채널 수.
        # 하지만 여기서는 Up 블록 내부에서 ConvTranspose2D가 채널 수를 조절하므로,
        # down4의 출력 채널이 up1의 in_channels로 들어감.
        self.up1 = Up(m * 512 // div_factor, m * 512 // div_factor, p=0, film=film, attn=False, bilinear=self.bilinear)
        # up1의 출력 채널(m*512) + down3의 출력 채널(m*512) = m*1024
        self.up2 = Up(m * (512 + 512) // div_factor, m * 256 // div_factor, p=0, film=film, attn=False, bilinear=self.bilinear)
        # up2의 출력 채널(m*256) + down2의 출력 채널(m*256) = m*512
        self.up3 = Up(m * (256 + 256) // div_factor, m * 128 // div_factor, p=0, film=film, attn=False, bilinear=self.bilinear)
        # up3의 출력 채널(m*128) + down1의 출력 채널(m*128) = m*256
        self.up4 = Up(m * (128 + 128) // div_factor, m * 64 // div_factor, p=0, film=film, attn=False, bilinear=self.bilinear)

        # 최종 출력 레이어 (1x1 컨볼루션)
        # 입력 채널: up4의 출력 채널(m*64) + 초기 입력 x1의 채널(in_channels)
        self.out = nn.Conv2d(
            m * 64 // div_factor + in_channels, # 이 부분은 U-Net 구조상 skip connection이 첫 입력과도 연결되는 형태여야 함.
                                             # 현재 Up 클래스는 x2와 x1을 합치므로, up4의 출력은 (m*64 + m*128) // div_factor 이거나,
                                             # Up 클래스에서 cat([x2, x1]) 후 self.conv를 통과한 채널 수 (out_channels)가 됨.
                                             # 따라서 self.up4.conv.bn_2.num_features 또는 self.up4.conv.conv_2.out_channels 가 더 정확.
                                             # 현재는 m * 64 // div_factor 로 되어 있어, up4의 out_channels를 의미.
                                             # 하지만 Up클래스의 forward는 cat([x2,x1])을 반환하므로, up4의 출력 채널은
                                             # x1(up4.conv의 out_channels = m*64//div_factor) + x2(down1의 out_channels = m*128//div_factor)
                                             # 즉, m*(64+128)//div_factor 가 됨.
                                             # 만약 outc의 입력이 x1 (초기 입력)과 up4의 출력을 합치는 것이라면,
                                             # self.up4의 out_channels + in_channels 가 되어야 함.
                                             # 현재 코드는 up4의 출력 채널 수(m*64//div_factor)와 초기 입력 채널 수를 더함.
            out_channels,
            kernel_size=1,
            bias=False, # CylindricalConvTranspose2D 에서 bias를 직접 관리하므로 여기서는 False
        )

    def forward(self, x, film_index=None):
        """U-Net 모델의 순전파 연산을 정의합니다."""

        x1 = x.contiguous()  # 입력 (skip connection 용)

        # 인코더 경로
        x2 = self.down1(x1, film_index=film_index)
        x3 = self.down2(x2, film_index=film_index)
        x4 = self.down3(x3, film_index=film_index)
        x5 = self.down4(x4, film_index=film_index)  # 가장 깊은 특징 맵

        # 디코더 경로 (skip connection과 함께 업샘플링)
        # x5를 업샘플링하고 x4와 결합
        x_up1 = self.up1(x5, x4, film_index=film_index)
        # x_up1을 업샘플링하고 x3과 결합
        x_up2 = self.up2(x_up1, x3, film_index=film_index)
        # x_up2를 업샘플링하고 x2와 결합
        x_up3 = self.up3(x_up2, x2, film_index=film_index)
        # x_up3을 업샘플링하고 x1(초기 입력)과 결합
        x_up4 = self.up4(x_up3, x1, film_index=film_index)

        # 최종 출력 레이어
        out = self.out(x_up4)

        # 출력 텐서의 축 순서를 (배치, 높이, 너비, 채널)로 변경
        return out.permute(0, 2, 3, 1)
