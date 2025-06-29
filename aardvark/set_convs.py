# PyTorch 라이브러리 임포트
import torch
import torch.nn as nn


class convDeepSet(nn.Module):
    """
    ConvDeepSet 클래스.
    그리드화되지 않은 관측치를 그리드 표현으로 변환하거나 그 반대로 변환하는 데 사용됩니다.
    일종의 어텐션 메커니즘 또는 커널 기반 보간 방법으로 볼 수 있으며,
    입력 지점과 출력 지점 간의 거리에 기반하여 가중치를 계산하고,
    이 가중치를 사용하여 입력값을 출력 그리드 또는 지점으로 변환합니다.
    """

    def __init__(
        self,
        init_ls,  # 초기 길이 스케일(length scale). 가중치 계산 시 사용되는 커널의 폭을 결정.
        mode,  # 연산 모드 ("OffToOn", "OnToOn", "OnToOff").
        device,  # 사용할 장치 (e.g., "cuda", "cpu").
        density_channel=True,  # 밀도 채널 사용 여부.
        step=0.25,  # 사용되지 않는 파라미터로 보임.
        grid=False,  # 사용되지 않는 파라미터로 보임.
    ):
        super().__init__()
        # 초기 길이 스케일을 학습 가능한 파라미터로 설정.
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls], device=device)) # device 인자 추가
        self.grid = grid # 사용되지 않음
        self.step = step # 사용되지 않음
        self.density_channel = density_channel
        self.mode = mode

        self.init_ls.requires_grad = True  # 길이 스케일이 학습 중에 업데이트되도록 설정.
        self.device = device

    def compute_weights(self, x1, x2):
        """
        두 좌표 집합 x1, x2 간의 가중치를 계산합니다.
        가중치는 두 점 사이의 거리에 대한 가우시안 커널(RBF 커널)을 사용하여 계산됩니다.
        길이 스케일(self.init_ls)은 커널의 폭을 조절합니다.

        Args:
            x1 (torch.Tensor): 첫 번째 좌표 집합 (예: 입력 지점 좌표).
            x2 (torch.Tensor): 두 번째 좌표 집합 (예: 출력 그리드 좌표).

        Returns:
            torch.Tensor: 계산된 가중치 텐서.
        """
        # 두 좌표 집합 간의 쌍별 제곱 거리(pairwise squared distances) 계산.
        # x1, x2의 마지막 차원을 확장하여 브로드캐스팅이 가능하도록 함.
        dists2 = self.pw_dists2(x1.unsqueeze(-1), x2.unsqueeze(-1))

        # 가우시안 커널을 사용하여 가중치 계산.
        # self.init_ls.to(x1.device)는 길이 스케일을 x1과 동일한 장치로 이동시킴.
        d = torch.exp((-0.5 * dists2) / (self.init_ls.to(x1.device)) ** 2)
        return d

    def pw_dists2(self, a, b):
        """
        두 텐서 a, b 간의 쌍별 제곱 거리를 효율적으로 계산합니다.
        (a-b)^2 = a^2 + b^2 - 2ab 공식을 활용.

        Args:
            a (torch.Tensor): 첫 번째 텐서. shape: (..., N, 1) 또는 (..., N, D) 등
            b (torch.Tensor): 두 번째 텐서. shape: (..., M, 1) 또는 (..., M, D) 등

        Returns:
            torch.Tensor: 쌍별 제곱 거리 텐서. shape: (..., N, M)
        """
        # 각 텐서 요소의 제곱합 계산 (L2 노름의 제곱).
        # keepdim=True 대신 적절한 None 인덱싱으로 차원 유지 또는 확장.
        norms_a = torch.sum(a**2, axis=-1)[..., :, None]  # (..., N, 1)
        norms_b = torch.sum(b**2, axis=-1)[..., None, :]  # (..., 1, M)

        # -2 * a @ b^T 계산. b.permute는 b의 마지막 두 차원을 전치.
        # matmul은 배치 행렬 곱셈을 지원.
        # a의 shape이 (B, N, D), b의 shape이 (B, M, D)라면, b.permute는 (B, D, M)이 되고,
        # 결과는 (B, N, M)이 됨.
        # 현재 코드에서는 x1.unsqueeze(-1) 등으로 인해 D=1인 경우로 보임.
        return norms_a + norms_b - 2 * torch.matmul(a, b.permute(0, 2, 1))

    def forward(self, x_in, wt, x_out):
        """
        ConvDeepSet의 순전파 연산을 수행합니다.

        Args:
            x_in (list of torch.Tensor): 입력 좌표 [lon, lat].
                                         "OffToOn" 모드: (배치, N_in_points)
                                         "OnToOn", "OnToOff" 모드: (배치, N_in_grid_dim)
            wt (torch.Tensor): 입력 데이터 값.
                               "OffToOn" 모드: (배치, 채널, N_in_points)
                               "OnToOn", "OnToOff" 모드: (배치, 채널, H_in, W_in)
            x_out (list of torch.Tensor): 출력 좌표 [lon, lat].
                                          "OffToOn", "OnToOn" 모드: (배치, N_out_grid_dim)
                                          "OnToOff" 모드: (배치, N_out_points)

        Returns:
            torch.Tensor: 변환된 출력 텐서.
        """

        # 밀도 채널 추가: 관측이 있는 곳은 1, 없는 곳(NaN)은 0으로 표시.
        # wt의 첫 번째 채널을 기준으로 생성.
        density_channel_val = torch.ones_like(wt[:, 0:1, ...]) # 변수명 변경
        density_channel_val[torch.isnan(wt[:, 0:1, ...])] = 0

        # 밀도 채널을 원래 가중치(wt)에 결합하고 NaN 값을 0으로 처리.
        wt_processed = torch.cat([density_channel_val, wt], dim=1) # 변수명 변경
        wt_processed[torch.isnan(wt_processed)] = 0

        if self.mode == "OffToOn":
            # 경우 1: 그리드화되지 않은 데이터(관측 지점)를 그리드 표현으로 변환.
            # x_in: [lon_obs, lat_obs], wt: obs_values, x_out: [lon_grid, lat_grid]

            # 입력 좌표에서 NaN이 아닌 유효한 부분에 대한 마스크 생성.
            in_lon_mask = ~torch.isnan(x_in[0])
            in_lat_mask = ~torch.isnan(x_in[1])

            # NaN 좌표를 0으로 대체 (가중치 계산 시 문제 방지).
            # 실제로는 마스크를 통해 가중치 계산에서 제외됨.
            x_in_processed = [x_in[0].clone(), x_in[1].clone()] # 원본 x_in 변경 방지
            x_in_processed[0][~in_lon_mask] = 0
            x_in_processed[1][~in_lat_mask] = 0

            # 입력 좌표와 출력 그리드 좌표 간의 가중치 계산.
            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in_processed, x_out)]

            # 유효하지 않은 입력 좌표에 대한 가중치를 0으로 만듦.
            # unsqueeze(-1)는 마스크를 브로드캐스팅 가능하게 확장.
            ws[0] = ws[0] * in_lon_mask.unsqueeze(-1).int()
            ws[1] = ws[1] * in_lat_mask.unsqueeze(-1).int()

            # Einstein summation을 사용하여 가중 합산 수행.
            # ...c w, ...w x, ...w y -> ...c xy
            # c: 채널, w: 입력 관측 지점, x: 출력 그리드 경도, y: 출력 그리드 위도
            ee = torch.einsum("...cw,...wx,...wy->...cxy", wt_processed, ws[0], ws[1])

        elif self.mode == "OnToOn":
            # 경우 2: 한 그리드 표현을 다른 그리드 표현으로 변환.
            # x_in: [lon_grid_in, lat_grid_in], wt: grid_values_in, x_out: [lon_grid_out, lat_grid_out]

            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out)]
            # ...c wh, ...w x, ...h y -> ...c xy
            # c: 채널, w: 입력 그리드 경도, h: 입력 그리드 위도, x: 출력 그리드 경도, y: 출력 그리드 위도
            ee = torch.einsum("...cwh,...wx,...hy->...cxy", wt_processed, ws[0], ws[1])

        elif self.mode == "OnToOff":
            # 경우 3: 그리드 표현을 그리드화되지 않은 지점(관측 지점)의 예측값으로 변환.
            # x_in: [lon_grid, lat_grid], wt: grid_values, x_out: [lon_obs, lat_obs]

            out_lon_mask = ~torch.isnan(x_out[0])
            out_lat_mask = ~torch.isnan(x_out[1])

            x_out_processed = [x_out[0].clone(), x_out[1].clone()] # 원본 x_out 변경 방지
            x_out_processed[0][~out_lon_mask] = 0
            x_out_processed[1][~out_lat_mask] = 0

            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out_processed)]

            # 유효하지 않은 출력 좌표에 대한 가중치를 0으로 만듦.
            # unsqueeze(-2)는 마스크를 브로드캐스팅 가능하게 확장.
            ws[0] = ws[0] * out_lon_mask.unsqueeze(-2).int()
            ws[1] = ws[1] * out_lat_mask.unsqueeze(-2).int()

            # ...c wh, ...w x, ...h x -> ...c x (x_out이 1D 좌표 리스트일 경우)
            # 또는 ...c wh, ...w x, ...h y -> ...c xy (x_out이 2D 좌표 리스트일 경우)
            # 현재 einsum은 x_out의 각 차원이 독립적으로 처리됨을 가정.
            # ...cwh,...wx,...hx->...cx 에서 x_out의 두 번째 차원(위도) 가중치가 ws[1]로,
            # 이는 경도와 위도 가중치를 곱하는 형태.
            # 만약 x_out이 (batch, n_points) 형태의 1D 좌표 리스트라면,
            # ws[0]은 (batch, n_grid_lon, n_points), ws[1]은 (batch, n_grid_lat, n_points)
            # torch.einsum("...cwh,wob,hob->...cob", wt, ws_lon, ws_lat) 형태가 더 적절할 수 있음 (o: output points)
            # 현재 코드는 x_out의 각 좌표 차원에 대해 독립적인 가중치를 적용하고,
            # 그 결과를 합치는 방식으로 동작하는 것으로 보임.
            # ...cwh,...wx,...hx->...cx 의 의미는
            # c: 채널, w: 입력 그리드 경도, h: 입력 그리드 위도, x: 출력 지점
            # 이는 x_out의 경도와 위도 좌표가 동일한 인덱스를 공유한다고 가정 (즉, x_out은 N개의 (lon,lat) 쌍).
            # 이 경우 ws[0]은 입력 경도 그리드와 출력 지점 경도 간의 가중치,
            # ws[1]은 입력 위도 그리드와 출력 지점 위도 간의 가중치.
            # einsum은 이 두 가중치를 곱하고 입력값(wt)과 합산.
            ee = torch.einsum("...cwh,...wx,...hx->...cx", wt_processed, ws[0], ws[1]) # hx -> hy가 더 일반적이나, x_out이 1D 지점 리스트일 수 있음

        else: # 알 수 없는 모드일 경우 오류 발생
            raise ValueError(f"알 수 없는 모드: {self.mode}")

        # 밀도 채널을 사용하여 정규화.
        if self.density_channel:
            # 첫 번째 채널(밀도 채널)과 나머지 채널(값 채널) 분리.
            # 값 채널을 밀도 채널로 나누어 정규화.
            # clamp를 사용하여 0으로 나누는 것을 방지하고 극단적인 값 제한.
            density = ee[:, 0:1, ...]
            values = ee[:, 1:, ...]
            normalized_values = values / torch.clamp(density, min=1e-6, max=1e5)
            ee = torch.cat([density, normalized_values], dim=1)
            return ee
        else:
            # 밀도 채널을 사용하지 않는 경우, 모든 채널을 밀도 채널(첫 번째 채널)로 나누어 정규화.
            # 이 경우, 입력 wt에 밀도 정보가 없거나 다른 방식으로 처리됨을 가정.
            density = ee[:, 0:1, ...] # 여전히 첫번째 채널을 밀도로 사용
            values = ee[:, 1:, ...]
            normalized_values = values / torch.clamp(density, min=1e-6, max=1e5)
            return normalized_values
