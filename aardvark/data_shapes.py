# 데이터 파일의 형태(shape) 정보를 정의하는 모듈입니다.
# 데이터는 접근 속도를 위해 메모리 맵(memmap) 형태로 저장되며,
# 이 파일들은 해당 메모리 맵 파일을 로드하는 데 필요한 형태 정보를 제공합니다.

# 기후학(Climatology) 데이터의 형태.
# (시간대_구분, 연중일, 변수_수, 경도_수, 위도_수) 등으로 해석될 수 있음.
# 예: (하루 4개 시간대, 366일, 24개 변수, 240 경도점, 121 위도점)
CLIMATOLOGY_SHAPE = (4, 366, 24, 240, 121)

# ICOADS(International Comprehensive Ocean-Atmosphere Data Set) 데이터 형태.
# ICOADS_Y_SHAPE: 관측값 데이터의 형태. (시간_스텝_수, 변수_수, 관측_지점_수) 등으로 해석 가능.
ICOADS_Y_SHAPE = (33601, 5, 12000)
# ICOADS_X_SHAPE: 관측 위치(경도, 위도) 데이터의 형태. (시간_스텝_수, 좌표_종류_수, 관측_지점_수)
ICOADS_X_SHAPE = (33601, 2, 12000)

# IGRA(Integrated Global Radiosonde Archive) 데이터 형태.
# IGRA_Y_SHAPE: 라디오존데 관측값 데이터의 형태. (시간_스텝_수, 변수_수, 관측소_수)
IGRA_Y_SHAPE = (33604, 24, 1375)
# IGRA_X_SHAPE: 라디오존데 관측소 위치 데이터의 형태. (관측소_수, 좌표_종류_수)
IGRA_X_SHAPE = (1375, 2) # IGRA_Y_SHAPE의 마지막 차원과 일치해야 함

# 다양한 위성 데이터의 형태.
# 일반적으로 (시간_스텝_수, 위도_수, 경도_수, 채널_수) 또는 유사한 순서로 구성됨.

# AMSU-A(Advanced Microwave Sounding Unit-A) 데이터 형태.
AMSUA_Y_SHAPE = (21916, 180, 360, 13)
# AMSU-B(Advanced Microwave Sounding Unit-B) 데이터 형태.
AMSUB_Y_SHAPE = (21916, 360, 181, 12)
# ASCAT(Advanced Scatterometer) 데이터 형태.
ASCAT_Y_SHAPE = (21913, 360, 181, 17)
# HIRS(High-resolution Infrared Radiation Sounder) 데이터 형태.
HIRS_Y_SHAPE = (21913, 360, 181, 26)
# GRIDSAT (격자화된 위성) 데이터 형태.
GRIDSAT_Y_SHAPE = (48211, 2, 514, 200) # (시간, 채널?, 위도, 경도) - 순서 확인 필요
# IASI(Infrared Atmospheric Sounding Interferometer) 데이터 형태.
IASI_Y_SHAPE = (23373, 360, 181, 52)


def get_hadisd_shape(mode, var=None): # var 파라미터 추가 (train 모드에서 사용됨)
    """
    HadISD(Hadley Centre Integrated Surface Database) 데이터 배열의 형태를 반환합니다.
    데이터 모드(train, val, test) 및 변수(var)에 따라 형태가 달라질 수 있습니다.

    Args:
        mode (str): 데이터 모드 ("train", "val", "test" 등).
        var (str, optional): 변수 이름 ("tas", "tds", "psl", "u", "v").
                             'train' 모드일 때 필요합니다.

    Returns:
        tuple: HadISD 데이터의 형태. (시간_스텝_수, 관측소_수_또는_특성_수)
    """

    if mode != "train":
        # 학습 모드가 아닐 경우 (예: val, test) 두 번째 차원 크기는 415로 고정.
        # 이는 특정 수의 검증/테스트 관측소 또는 특성을 의미할 수 있음.
        dim_1 = 415
    else:
        # 학습 모드일 경우, 변수별로 두 번째 차원 크기가 다름.
        if var is None:
            raise ValueError("학습 모드에서는 'var' 인자가 반드시 필요합니다.")
        var_dict = {
            "tas": 8719,  # 기온(tas) 변수의 경우
            "tds": 8617,  # 이슬점 온도(tds) 변수의 경우
            "psl": 8016,  # 해면 기압(psl) 변수의 경우
            "u": 8721,    # U-성분 바람(u) 변수의 경우
            "v": 8721     # V-성분 바람(v) 변수의 경우
        }
        if var not in var_dict:
            raise ValueError(f"알 수 없는 변수명: {var}. 유효한 변수명: {list(var_dict.keys())}")
        dim_1 = var_dict[var]
    # 첫 번째 차원(시간 스텝 수)은 106652로 고정.
    return (106652, dim_1)
