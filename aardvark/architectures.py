# PyTorch 라이브러리 임포트
import torch
import torch.nn as nn

# 현재 디렉토리의 utils 모듈에서 모든 것을 임포트
# utils 모듈에 어떤 함수나 클래스가 있는지 확인 필요
from utils import *


class MLP(nn.Module):
    """
    다층 퍼셉트론 (Multi-layer Perceptron) 모델 클래스.
    간단한 완전 연결 신경망을 구성합니다.
    """

    def __init__(
        self,
        in_channels,  # 입력 특징(채널)의 수
        out_channels,  # 출력 특징(채널)의 수
        h_channels=64,  # 은닉층의 특징(채널) 수
        h_layers=4,  # 은닉층의 수
    ):

        super().__init__()  # 부모 클래스(nn.Module)의 초기화 함수 호출

        def hidden_block(h_channels_local):  # 내부 함수로 은닉 블록 정의, h_channels 이름 충돌 방지
            """단일 은닉층 블록을 생성합니다."""
            h = nn.Sequential(  # 순차적인 계층들을 그룹화
                nn.Linear(h_channels_local, h_channels_local),  # 선형 변환 (완전 연결층)
                nn.ReLU(),  # ReLU 활성화 함수
            )
            return h

        # MLP 모델의 전체 계층 구성
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),  # 입력층 -> 첫 번째 은닉층
            nn.ReLU(),  # 활성화 함수
            # 리스트 컴프리헨션을 사용하여 지정된 수만큼 은닉 블록 추가
            *[hidden_block(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels)  # 마지막 은닉층 -> 출력층
        )

    def forward(self, x):
        """
        모델의 순전파 연산을 정의합니다.

        Args:
            x (torch.Tensor): 입력 텐서.

        Returns:
            torch.Tensor: 모델의 출력 텐서.
        """
        return self.mlp(x)  # 구성된 MLP를 통해 입력 x를 전달
