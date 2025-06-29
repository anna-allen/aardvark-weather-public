# 이 파일은 Nguyen, Tung, et al. "ClimaX: A foundation model
# for weather and climate." arXiv preprint arXiv:2301.10343 (2023) 논문에서 파생되었습니다.
# 해당 프로젝트 코드는 https://github.com/microsoft/ClimaX 에서 확인할 수 있습니다.

# functools 모듈에서 lru_cache 데코레이터 임포트 (함수 결과 캐싱에 사용)
from functools import lru_cache

# NumPy, PyTorch 라이브러리 임포트
import numpy as np
import torch
import torch.nn as nn
# timm (PyTorch Image Models) 라이브러리에서 ViT 관련 클래스 및 함수 임포트
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

# 현재 디렉토리의 architectures 모듈에서 MLP 클래스 임포트
from architectures import MLP


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    2D 사인/코사인 위치 임베딩을 생성합니다.
    ViT (Vision Transformer)에서 패치들의 위치 정보를 인코딩하는 데 사용됩니다.

    Args:
        embed_dim (int): 임베딩 차원.
        grid_size_h (int): 그리드 높이 (패치 수 기준).
        grid_size_w (int): 그리드 너비 (패치 수 기준).
        cls_token (bool, optional): CLS 토큰 사용 여부. True이면 CLS 토큰용 제로 임베딩을 추가합니다.

    Returns:
        np.ndarray: 생성된 2D 위치 임베딩.
    """
    # 높이 및 너비 방향으로 그리드 좌표 생성
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 너비 우선, 높이 다음 순서로 그리드 생성
    grid = np.stack(grid, axis=0)  # (2, grid_size_h, grid_size_w) 형태로 스택

    # 그리드 형태 변경 및 실제 위치 임베딩 생성 함수 호출
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # CLS 토큰 사용 시, 맨 앞에 제로 벡터 추가
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    주어진 그리드로부터 2D 사인/코사인 위치 임베딩을 생성합니다.
    높이와 너비 방향 각각에 대해 1D 위치 임베딩을 생성하고 결합합니다.

    Args:
        embed_dim (int): 전체 임베딩 차원. 짝수여야 합니다.
        grid (np.ndarray): (2, 1, grid_size_h, grid_size_w) 형태의 그리드 좌표.

    Returns:
        np.ndarray: 생성된 2D 위치 임베딩.
    """
    assert embed_dim % 2 == 0, "임베딩 차원은 짝수여야 합니다."

    # 높이 방향 임베딩 (전체 차원의 절반 사용)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    # 너비 방향 임베딩 (전체 차원의 절반 사용)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    # 높이와 너비 임베딩을 특징 차원을 따라 결합
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    주어진 1D 위치 배열로부터 1D 사인/코사인 위치 임베딩을 생성합니다.
    Transformer에서 사용되는 고정된 위치 인코딩 방식입니다.

    Args:
        embed_dim (int): 임베딩 차원. 짝수여야 합니다.
        pos (np.ndarray): 1D 위치 배열.

    Returns:
        np.ndarray: 생성된 1D 위치 임베딩.
    """
    assert embed_dim % 2 == 0, "임베딩 차원은 짝수여야 합니다."

    # 주파수 계산을 위한 오메가 값 생성
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # 10000은 일반적으로 사용되는 스케일링 팩터

    pos = pos.reshape(-1)  # 위치 배열을 1차원으로 펼침
    out = np.einsum("m,d->md", pos, omega)  # 위치와 오메가의 외적 계산

    # 사인 및 코사인 함수 적용
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    # 사인과 코사인 임베딩을 번갈아가며 결합 (또는 단순히 이어붙임)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def interpolate_pos_embed(model, checkpoint_model, new_size=(64, 128)):
    """
    사전 학습된 모델의 위치 임베딩을 새로운 입력 크기에 맞게 보간합니다.
    모델의 이미지 크기나 패치 크기가 변경되었을 때 사용됩니다.

    Args:
        model (nn.Module): 현재 모델 인스턴스.
        checkpoint_model (dict): 로드된 체크포인트의 state_dict.
        new_size (tuple, optional): 새로운 이미지 크기 (높이, 너비).
    """
    if "net.pos_embed" in checkpoint_model:  # 체크포인트에 위치 임베딩이 있는지 확인
        pos_embed_checkpoint = checkpoint_model["net.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]  # 임베딩 차원
        orig_num_patches = pos_embed_checkpoint.shape[-2]  # 원본 패치 수
        patch_size = model.patch_size  # 현재 모델의 패치 크기

        # 원본 이미지의 높이/너비 비율을 기반으로 원본 패치 그리드 크기 추정
        # 이 부분은 이미지의 가로세로 비율(w_h_ratio)이 고정되어 있다고 가정함 (여기서는 2)
        w_h_ratio = 2
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w) # 원본 패치 그리드 크기

        # 새로운 패치 그리드 크기 계산
        new_patch_grid_size = (new_size[0] // patch_size, new_size[1] // patch_size)

        # 원본과 새로운 패치 그리드 크기가 다를 경우 보간 수행
        if orig_size[0] != new_patch_grid_size[0] or orig_size[1] != new_patch_grid_size[1]: # 조건 수정
            print(
                f"위치 임베딩 보간: ({orig_size[0]}x{orig_size[1]}) -> ({new_patch_grid_size[0]}x{new_patch_grid_size[1]})"
            )
            # 위치 임베딩을 (배치, 채널, 높이, 너비) 형태로 변경
            pos_tokens = pos_embed_checkpoint.reshape(
                -1, orig_size[0], orig_size[1], embedding_size
            ).permute(0, 3, 1, 2)
            # Bicubic 보간을 사용하여 크기 조정
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_patch_grid_size[0], new_patch_grid_size[1]),
                mode="bicubic",
                align_corners=False,
            )
            # 원래 형태로 되돌리고 체크포인트 모델에 업데이트
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2) # (배치, 총패치수, 임베딩차원)
            checkpoint_model["net.pos_embed"] = new_pos_tokens


def interpolate_channel_embed(checkpoint_model, new_len):
    """
    사전 학습된 모델의 채널 임베딩 길이를 새로운 길이에 맞게 조정합니다.
    입력 변수의 수가 변경되었을 때 사용될 수 있습니다.

    Args:
        checkpoint_model (dict): 로드된 체크포인트의 state_dict.
        new_len (int): 새로운 채널 임베딩 길이 (새로운 변수 수).
    """
    if "net.channel_embed" in checkpoint_model: # 체크포인트에 채널 임베딩이 있는지 확인
        channel_embed_checkpoint = checkpoint_model["net.channel_embed"]
        old_len = channel_embed_checkpoint.shape[1] # 원본 채널 임베딩 길이
        if new_len <= old_len: # 새로운 길이가 기존보다 작거나 같으면 앞부분만 사용
            checkpoint_model["net.channel_embed"] = channel_embed_checkpoint[
                :, :new_len
            ]
        # else: 새로운 길이가 더 길 경우 처리 방법이 정의되지 않음 (예: 제로 패딩 또는 재학습)


class ViT(nn.Module):
    """
    Vision Transformer (ViT) 모델 클래스.
    이미지를 여러 패치로 나누고, 각 패치를 임베딩하여 트랜스포머 인코더에 입력합니다.
    기상 및 기후 데이터 처리를 위해 ClimaX 논문의 아이디어를 일부 차용합니다.
    """

    def __init__(
        self,
        in_channels,  # 입력 이미지의 채널 수 (또는 변수 수)
        out_channels,  # 최종 출력 채널 수 (예측 변수 수)
        h_channels,  # 내부 임베딩 차원 (embed_dim)
        img_size=[256, 128],  # 입력 이미지 크기 [높이, 너비]
        patch_size=8,  # 각 패치의 크기
        depth=24,  # 트랜스포머 블록의 수
        decoder_depth=4,  # 최종 예측 헤드의 깊이 (MLP 레이어 수)
        num_heads=16,  # 멀티헤드 어텐션의 헤드 수
        mlp_ratio=4.0,  # 트랜스포머 블록 내 MLP의 확장 비율
        drop_path=0.0,  # Stochastic Depth를 위한 드롭 경로 확률
        drop_rate=0.0,  # 드롭아웃 확률
        per_var_embedding=True,  # 변수별 패치 임베딩 사용 여부
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        # 기본 변수명 생성 (0, 1, 2, ... 순서)
        default_vars = [str(i) for i in range(in_channels)]
        self.default_vars = default_vars
        embed_dim = h_channels  # 내부 임베딩 차원을 h_channels로 통일
        self.per_var_embedding = per_var_embedding

        # 패치 임베딩 레이어 생성
        if self.per_var_embedding:
            # 각 변수(채널)에 대해 독립적인 PatchEmbed 레이어 사용
            self.token_embeds = nn.ModuleList(
                [
                    PatchEmbed(img_size, patch_size, 1, embed_dim) # 각 변수는 1채널로 처리
                    for _ in range(len(default_vars)) # 변수 수만큼 생성
                ]
            )
        else:
            # 모든 변수(채널)를 한 번에 처리하는 단일 PatchEmbed 레이어 사용
            # 또는 MLP를 통해 채널 수를 맞춘 후 PatchEmbed 적용 (아래 mlp 초기화 부분 참고)
            self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, in_channels, embed_dim)]
            )
        self.num_patches = self.token_embeds[0].num_patches # 총 패치 수

        # 변수 임베딩 및 변수-ID 매핑 생성
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        # 변수 집계를 위한 학습 가능한 쿼리 벡터
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        # 변수 집계용 멀티헤드 어텐션 레이어
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 위치 임베딩 파라미터 (학습 가능)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        # 리드 타임 임베딩을 위한 선형 레이어
        self.lead_time_embed = nn.Linear(1, embed_dim)

        self.out_dim = out_channels  # 최종 출력 변수 수
        self.pos_drop = nn.Dropout(p=drop_rate)  # 위치 임베딩 후 드롭아웃

        # Stochastic Depth를 위한 드롭 경로 확률 리스트 생성
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        # 트랜스포머 블록(인코더) 생성
        self.blocks = nn.ModuleList(
            [
                Block( # timm 라이브러리의 Block 사용
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i], # 각 블록마다 다른 드롭 경로 확률 적용
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)  # 트랜스포머 블록 후 Layer Normalization

        # 예측 헤드 (MLP 형태의 디코더)
        self.head = nn.ModuleList()
        for _ in range(decoder_depth): # 지정된 깊이만큼 선형 레이어와 GELU 활성화 함수 추가
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        # 최종 출력 레이어: 패치별로 각 출력 변수 값을 예측하도록 차원 설정
        self.head.append(nn.Linear(embed_dim, self.out_dim * patch_size**2))
        self.head = nn.Sequential(*self.head) # ModuleList를 Sequential 모델로 변환

        self.initialize_weights() # 가중치 초기화 함수 호출

        # per_var_embedding=False 일 때 사용할 MLP (입력 채널 수를 맞추기 위함일 수 있음)
        # in_channels=277 값의 출처 확인 필요 (특정 데이터셋의 채널 수일 가능성)
        if not self.per_var_embedding:
            self.mlp = MLP(in_channels=277, out_channels=256) # architectures.MLP 사용

    def initialize_weights(self):
        """모델의 가중치를 초기화합니다."""
        # 2D 사인/코사인 위치 임베딩으로 self.pos_embed 초기화
        pos_embed_val = get_2d_sincos_pos_embed( # 변수명 변경
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size), # 패치 그리드 높이
            int(self.img_size[1] / self.patch_size), # 패치 그리드 너비
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_val).float().unsqueeze(0))

        # 1D 사인/코사인 위치 임베딩으로 self.var_embed 초기화
        var_embed_val = get_1d_sincos_pos_embed_from_grid( # 변수명 변경
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed_val).float().unsqueeze(0))

        # 패치 임베딩 레이어의 프로젝션 가중치 초기화 (Truncated Normal)
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # 나머지 레이어들에 대한 가중치 초기화 (아래 _init_weights 함수 적용)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """특정 타입의 레이어 가중치를 초기화하는 헬퍼 함수."""
        if isinstance(m, nn.Linear): # 선형 레이어의 경우
            trunc_normal_(m.weight, std=0.02) # 가중치를 Truncated Normal 분포로 초기화
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # 바이어스를 0으로 초기화
        elif isinstance(m, nn.LayerNorm): # LayerNorm 레이어의 경우
            nn.init.constant_(m.bias, 0) # 바이어스를 0으로 초기화
            nn.init.constant_(m.weight, 1.0) # 가중치(스케일)를 1로 초기화

    def create_var_embedding(self, dim):
        """변수 임베딩 파라미터와 변수-ID 매핑을 생성합니다."""
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.default_vars), dim), requires_grad=True
        )
        var_map = {}
        idx = 0
        for var_name in self.default_vars: # 변수명 var -> var_name
            var_map[var_name] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None) # 함수 결과를 캐싱하여 반복 계산 방지
    def get_var_ids(self, vars_tuple, device): # 인자명 vars -> vars_tuple (튜플임을 명시)
        """주어진 변수명 리스트에 대한 ID 리스트를 반환합니다."""
        ids = np.array([self.var_map[var_name] for var_name in vars_tuple]) # 변수명 var -> var_name
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb_param, vars_tuple): # 인자명 var_emb -> var_emb_param, vars -> vars_tuple
        """주어진 변수명 리스트에 해당하는 변수 임베딩을 반환합니다."""
        ids = self.get_var_ids(vars_tuple, var_emb_param.device)
        return var_emb_param[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        패치 시퀀스를 다시 이미지 형태로 복원합니다. (ViT의 디코딩 과정)

        Args:
            x (torch.Tensor): (배치, 총패치수, 패치별특징수) 형태의 텐서.
            h (int, optional): 복원될 이미지의 패치 그리드 높이. None이면 기본값 사용.
            w (int, optional): 복원될 이미지의 패치 그리드 너비. None이면 기본값 사용.

        Returns:
            torch.Tensor: (배치, 채널수, 높이, 너비) 형태의 이미지 텐서.
        """
        p = self.patch_size  # 패치 크기
        c = self.out_dim  # 출력 채널 수 (예측 변수 수)
        # 복원될 이미지의 패치 그리드 높이/너비 결정
        h_patch_grid = self.img_size[0] // p if h is None else h // p # 변수명 변경
        w_patch_grid = self.img_size[1] // p if w is None else w // p # 변수명 변경
        assert h_patch_grid * w_patch_grid == x.shape[1], "패치 수 불일치"

        # (배치, H_patch, W_patch, patch_h, patch_w, 채널) 형태로 변경
        x = x.reshape(shape=(x.shape[0], h_patch_grid, w_patch_grid, p, p, c))
        # 축 순서 변경: (배치, 채널, H_patch, patch_h, W_patch, patch_w)
        x = torch.einsum("nhwpqc->nchpwq", x) # n:배치, h:H_patch, w:W_patch, p:patch_h, q:patch_w, c:채널
        # 최종 이미지 형태로 복원: (배치, 채널, 높이, 너비)
        imgs = x.reshape(shape=(x.shape[0], c, h_patch_grid * p, w_patch_grid * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        변수별로 임베딩된 특징들을 집계하여 단일 특징 벡터로 만듭니다.
        MultiheadAttention을 사용하여 변수들 간의 관계를 학습하고 중요한 정보를 추출합니다.

        Args:
            x (torch.Tensor): (배치, 변수수, 패치수, 임베딩차원) 형태의 텐서.

        Returns:
            torch.Tensor: (배치, 패치수, 임베딩차원) 형태의 집계된 특징 텐서.
        """
        b, num_vars, l, dim_embed = x.shape # 변수명 변경 및 명시
        x = torch.einsum("bvld->blvd", x)  # (배치, 패치수, 변수수, 임베딩차원)으로 변경
        x = x.flatten(0, 1)  # (배치 * 패치수, 변수수, 임베딩차원)으로 펼침

        # 학습 가능한 var_query를 반복하여 배치 크기에 맞춤
        var_query_repeated = self.var_query.repeat_interleave(x.shape[0], dim=0) # 변수명 변경
        # MultiheadAttention 수행 (var_query가 query, x가 key/value)
        x, _ = self.var_agg(var_query_repeated, x, x)
        x = x.squeeze(dim=1) # (배치 * 패치수, 1, 임베딩차원) -> (배치 * 패치수, 임베딩차원)

        # 원래 배치 및 패치 차원으로 복원
        x = x.unflatten(dim=0, sizes=(b, l)) # (배치, 패치수, 임베딩차원)
        return x

    def mlp_embedding(self, x):
        """
        MLP 임베딩 함수 (현재는 아무 동작도 하지 않음).
        per_var_embedding=False일 때 사용될 수 있음.
        """
        # 이 함수는 self.mlp를 사용하여 채널 임베딩을 수행해야 할 것으로 보임.
        # 예: return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return # 현재는 아무것도 반환하지 않음

    def forward_encoder(self, x_input, lead_times, variables_tuple): # 변수명 변경
        """ViT 인코더의 순전파 과정을 정의합니다."""

        # variables가 리스트면 튜플로 변환 (get_var_ids 캐싱을 위함)
        if isinstance(variables_tuple, list):
            variables_tuple = tuple(variables_tuple)

        x_processed = x_input # 변수명 변경

        if self.per_var_embedding: # 변수별 임베딩 사용
            embeds = []
            var_ids = self.get_var_ids(variables_tuple, x_processed.device)
            for i in range(len(var_ids)):
                var_id = var_ids[i] # 변수명 변경
                # 각 변수에 해당하는 채널을 선택하여 해당 PatchEmbed 레이어 통과
                embeds.append(self.token_embeds[var_id](x_processed[:, i : i + 1])) # x_processed의 채널이 변수 순서와 일치한다고 가정
            x_processed = torch.stack(embeds, dim=1)  # (배치, 변수수, 패치수, 임베딩차원)

            # 변수 임베딩 추가
            var_embed_selected = self.get_var_emb(self.var_embed, variables_tuple) # 변수명 변경
            x_processed = x_processed + var_embed_selected.unsqueeze(2) # 패치 차원 추가하여 브로드캐스팅

            # 변수 집계
            x_processed = self.aggregate_variables(x_processed)
        else: # 단일 패치 임베딩 사용
            # MLP를 사용하여 입력 채널 수 조정 (예: 277 -> 256)
            # x_processed의 현재 shape은 (B, C_in, H, W)
            # MLP는 (B, H, W, C_in)을 기대하므로 permute 필요
            x_processed = self.mlp(x_processed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_processed = self.token_embeds[0](x_processed) # 단일 PatchEmbed 통과

        # 위치 임베딩 추가
        x_processed = x_processed + self.pos_embed

        # 리드 타임 임베딩 추가
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1).float()) # float 타입 명시
        lead_time_emb = lead_time_emb.unsqueeze(1) # 패치 차원 추가
        x_processed = x_processed + lead_time_emb

        x_processed = self.pos_drop(x_processed) # 드롭아웃

        # 트랜스포머 블록 통과
        for blk in self.blocks:
            x_processed = blk(x_processed)
        x_processed = self.norm(x_processed) # Layer Normalization

        return x_processed

    def forward(self, x_input, lead_times=None, film_index=None): # film_index는 현재 사용되지 않음
        """ViT 모델의 전체 순전파 과정을 정의합니다."""

        # 리드 타임이 제공되지 않으면 기본값(1로 채워진 텐서) 사용
        if lead_times is None:
            # .cuda() 대신 x_input.device 사용 권장
            lead_times = torch.ones(x_input.shape[0], device=x_input.device).unsqueeze(-1)

        # 인코더 순전파
        # lead_times[:, 0]는 lead_times가 (배치, 1) 형태일 때 스칼라 값을 전달하기 위함
        out_transformers = self.forward_encoder(x_input, lead_times[:, 0], self.default_vars)
        # 예측 헤드 통과
        preds = self.head(out_transformers)
        # 패치 시퀀스를 이미지 형태로 복원
        preds = self.unpatchify(preds)

        # 최종 출력의 축 순서를 (배치, 높이, 너비, 채널)로 변경
        return preds.permute(0, 2, 3, 1)
