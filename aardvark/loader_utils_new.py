# NumPy, PyTorch, Pandas 라이브러리 임포트
import numpy as np
import torch
import pandas as pd

# 경도/위도 스케일링 상수. 데이터를 [0, 1] 범위로 정규화하는 데 사용될 수 있음.
LATLON_SCALE_FACTOR = 360
# 연간 일수 상수. 시간 관련 특징 생성 시 사용 (예: 연중일 정규화). 윤년을 고려하여 366으로 설정된 것으로 보임.
DAYS_IN_YEAR = 366

# 각 데이터 소스별 일일 데이터 스케일 팩터.
# 예를 들어 ERA5 데이터가 6시간 간격이면 하루에 4개의 데이터 포인트가 있음.
DAILY_SCALE_FACTOR = {
    "ERA5": 4,  # ERA5 데이터는 하루 4번 (6시간 간격)
    "HADISD": 4,  # HADISD 데이터는 하루 4번
    "IR": 4,  # 적외선(IR) 데이터는 하루 4번
    "SOUNDER": 1,  # 사운더 데이터는 하루 1번
    "ICOADS": 1,  # ICOADS 데이터는 하루 1번
}

# 데이터 로딩 시 특정 시점의 인덱스를 찾기 위한 기준 날짜 목록.
# 주로 데이터셋의 시작/종료 지점 또는 중요한 이벤트 날짜를 나타냄.
date_list = [
    "1979-01-01",
    "1999-01-01",
    "1999-01-02",
    "2002-01-02",
    "2007-01-01",
    "2007-01-02",
    "2007-01-03",
    "2013-01-02",
    "2014-01-01",
    "2017-01-02",
    "2018-01-01",
    "2019-01-01",
    "2020-01-02",
    "2020-01-01",
    "2021-01-02",
    "2021-01-01",
]


def generate_offsets(date_list, dates):
    """
    주어진 전체 날짜 시퀀스(dates)에서 특정 날짜 목록(date_list)에 해당하는
    인덱스 오프셋을 생성합니다.

    Args:
        date_list (list): 오프셋을 찾고자 하는 특정 날짜 문자열 목록.
        dates (pd.DatetimeIndex): 전체 날짜 시퀀스.

    Returns:
        dict: 각 특정 날짜를 키로, 해당 날짜의 인덱스를 값으로 하는 딕셔너리.
              찾지 못한 경우 -1을 값으로 가짐.
    """
    offsets = {}
    for d_str in date_list:  # 변수명 d에서 d_str로 변경하여 명확성 증진
        try:
            # 문자열 날짜를 Timestamp 객체로 변환하여 비교 (dates가 DatetimeIndex일 경우)
            # 만약 dates가 문자열 리스트라면, 이 변환은 필요 없을 수 있음.
            # 여기서는 dates가 pd.date_range로 생성되므로 DatetimeIndex임.
            # np.where는 첫 번째 튜플에 인덱스 배열을 반환하므로 [0][0]으로 접근.
            offsets[d_str] = np.where(dates == pd.Timestamp(d_str))[0][0]
        except IndexError:  # 날짜를 찾지 못한 경우 IndexError 발생
            offsets[d_str] = -1
    return offsets


# 각 데이터 소스별로 특정 날짜들에 대한 인덱스 오프셋을 미리 계산.
# 이는 데이터 로딩 시 특정 시작 날짜로부터의 상대적 위치를 빠르게 찾는 데 사용됨.

# 초기 조건(Initial Condition) 데이터에 대한 오프셋
IC_OFFSETS = generate_offsets(
    date_list, pd.date_range("1999-01-02", "2021-12-31 18:00", freq="6H")
)

# AMSU-A 위성 데이터에 대한 오프셋
AMSUA_OFFSETS = generate_offsets(
    date_list, pd.date_range("2007-01-01", "2021-12-31 18:00", freq="6H")
)

# AMSU-B 위성 데이터에 대한 오프셋
AMSUB_OFFSETS = generate_offsets(
    date_list, pd.date_range("2007-01-01", "2021-12-31 18:00", freq="6H")
)

# ASCAT 위성 데이터에 대한 오프셋
ASCAT_OFFSETS = generate_offsets(
    date_list, pd.date_range("2007-01-01", "2021-12-31", freq="6H")
)

# ATMS 위성 데이터에 대한 오프셋 (1일 간격)
ATMS_OFFSETS = generate_offsets(
    date_list, pd.date_range("2013-01-02", "2021-12-31", freq="1D")
)

# ICOADS 해양 데이터에 대한 오프셋
ICOADS_OFFSETS = generate_offsets(
    date_list, pd.date_range("1999-01-01 06:00", "2021-12-31", freq="6H")
)

# IGRA 라디오존데 데이터에 대한 오프셋
IGRA_OFFSETS = generate_offsets(
    date_list, pd.date_range("1999-01-01 00:00", "2021-12-31 18:00", freq="6H")
)

# 일반 위성(SAT) 데이터에 대한 오프셋 (GRIDSAT 등)
SAT_OFFSETS = generate_offsets(
    date_list, pd.date_range("1990-01-01 00:00", "2021-12-31 18:00", freq="6H")
)

# HadISD 지표 관측 데이터에 대한 오프셋
HADISD_OFFSETS = generate_offsets(
    date_list, pd.date_range("1950-01-01 00:00", "2021-12-31 18:00", freq="6H")
)


def lon_to_0_360(x):
    """
    경도 값을 [0, 360) 범위로 변환합니다.
    예: -90 -> 270, 450 -> 90

    Args:
        x (float or np.ndarray): 변환할 경도 값.

    Returns:
        float or np.ndarray: [0, 360) 범위로 변환된 경도 값.
    """
    return (x + 360) % 360


def lat_to_m90_90(x):
    """
    위도 값을 [-90, 90] 범위로 변환 (또는 배열 순서를 뒤집습니다).
    PyTorch 텐서를 입력으로 받아 마지막 차원을 기준으로 뒤집습니다.
    주로 북쪽에서 남쪽으로 정렬된 위도 데이터를 남쪽에서 북쪽으로 변경하거나
    그 반대로 할 때 사용될 수 있습니다. 이름은 -90 ~ 90 범위를 암시하지만,
    실제 동작은 배열 뒤집기입니다.

    Args:
        x (torch.Tensor): 변환할 위도 값 또는 위도 관련 데이터 텐서.

    Returns:
        torch.Tensor: 마지막 차원이 뒤집힌 텐서.
    """
    return torch.flip(x, [-1])
