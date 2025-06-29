# collections 모듈에서 defaultdict 클래스 임포트 (딕셔너리 기본값 설정에 사용)
from collections import defaultdict
# NumPy 라이브러리 임포트 (주로 np.nan 사용)
import numpy as np
# PyTorch RNN 유틸리티에서 pad_sequence 함수 임포트 (가변 길이 시퀀스 패딩에 사용)
from torch.nn.utils.rnn import pad_sequence


def channels_to_2nd_dim(x):
    """
    입력 텐서의 채널 차원을 두 번째 차원으로 이동시킵니다.
    예를 들어, (배치, 높이, 너비, 채널) 형태의 텐서를 (배치, 채널, 높이, 너비) 형태로 변환합니다.
    PyTorch의 컨볼루션 레이어 등은 채널 차원이 두 번째에 오는 것을 기대합니다.

    Args:
        x (torch.Tensor): 입력 텐서. 채널 차원이 마지막에 있다고 가정합니다.

    Returns:
        torch.Tensor: 채널 차원이 두 번째로 이동된 텐서.
    """
    # x.dim() - 1은 마지막 차원(채널 차원)의 인덱스를 나타냅니다.
    # [0, x.dim() - 1] : 배치 차원과 채널 차원을 선택합니다.
    # list(range(1, x.dim() - 1)): 나머지 중간 차원들을 선택합니다.
    # 예: x.shape = (B, H, W, C) 라면, x.dim() = 4
    # permute 인자: (0, 3, 1, 2) -> (B, C, H, W)
    return x.permute(*([0, x.dim() - 1] + list(range(1, x.dim() - 1))))


def channels_to_final_dim(x):
    """
    입력 텐서의 두 번째 차원(채널 차원으로 가정)을 마지막 차원으로 이동시킵니다.
    예를 들어, (배치, 채널, 높이, 너비) 형태의 텐서를 (배치, 높이, 너비, 채널) 형태로 변환합니다.

    Args:
        x (torch.Tensor): 입력 텐서. 채널 차원이 두 번째에 있다고 가정합니다.

    Returns:
        torch.Tensor: 채널 차원이 마지막으로 이동된 텐서.
    """
    # [0] : 배치 차원을 선택합니다.
    # list(range(2, x.dim())): 채널 차원(인덱스 1)을 제외한 나머지 중간 차원들을 선택합니다.
    # [1] : 원래 채널 차원(인덱스 1)을 마지막으로 이동시킵니다.
    # 예: x.shape = (B, C, H, W) 라면, x.dim() = 4
    # permute 인자: (0, 2, 3, 1) -> (B, H, W, C)
    return x.permute(*([0] + list(range(2, x.dim())) + [1]))


def collate(tensor_list):
    """
    PyTorch DataLoader에서 사용될 수 있는 사용자 정의 `collate_fn` 함수입니다.
    딕셔너리 리스트를 입력으로 받아, 각 키에 해당하는 텐서들을 모아 패딩하고
    하나의 배치 딕셔너리로 만듭니다.
    주로 가변 길이의 시퀀스 데이터를 배치 처리할 때 유용합니다.

    Args:
        tensor_list (list of dict): 각 요소는 동일한 키를 가진 딕셔너리이며,
                                    각 키의 값은 패딩될 텐서입니다.

    Returns:
        defaultdict: 각 키에 대해 패딩된 텐서 배치를 값으로 가지는 딕셔너리.
    """
    out_dict = defaultdict(list) # 기본값이 빈 리스트인 딕셔너리 생성
    # 첫 번째 샘플의 키들을 기준으로 반복
    for k in tensor_list[0].keys():
        # 각 샘플에서 동일한 키(k)에 해당하는 텐서들을 리스트로 모음
        key_specific_tensors = [t[k] for t in tensor_list] # 변수명 변경

        # pad_sequence를 사용하여 텐서 리스트를 패딩함.
        # padding_value=np.nan: 패딩 값으로 NaN 사용 (주의: 모든 텐서 타입과 호환되지 않을 수 있음)
        # batch_first=True: 반환되는 텐서의 첫 번째 차원이 배치가 되도록 함.
        out_dict[k] = pad_sequence(
            key_specific_tensors, # 위에서 생성한 리스트 사용
            padding_value=float('nan'), # np.nan 대신 float('nan') 사용 (PyTorch 텐서와 호환성)
            batch_first=True,
        )
    return out_dict
