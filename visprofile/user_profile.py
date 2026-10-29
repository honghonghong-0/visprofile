"""단말에 보관되는 사용자 시각 선호 프로필과 갱신·압축 연산.

갱신은 과거 선호작 목록을 다시 읽지 않고 (현재 상태, 오늘의 선호작)만으로 수행되므로
연산량은 O(d), 상태 크기는 시청 수와 무관한 d차원 벡터이다.
"""
import numpy as np


class ProfileState:
    """한 사용자의 누적 평균(장기)·EMA(단기 가중) 프로필을 함께 유지한다."""

    def __init__(self, dim, alphas):
        self.sum = np.zeros(dim)
        self.count = 0
        self.alphas = tuple(alphas)
        self.ema = {a: None for a in self.alphas}

    @property
    def ready(self):
        return self.count > 0

    @property
    def mean(self):
        # p_t = p_{t-1} + (x̄_t - p_{t-1}) n_t / N_t 와 동치 (합과 개수로 보관)
        return self.sum / self.count

    def update(self, day_embeddings):
        """하루 동안 선호한 영화들의 임베딩 (n_t, d) 로 갱신."""
        if len(day_embeddings) == 0:
            return
        x_bar = day_embeddings.mean(0)
        self.sum += day_embeddings.sum(0)
        self.count += len(day_embeddings)
        for a in self.alphas:  # p_t = (1 - a) p_{t-1} + a x̄_t
            self.ema[a] = x_bar.copy() if self.ema[a] is None else (1 - a) * self.ema[a] + a * x_bar

    def snapshot(self):
        return self.mean, {a: self.ema[a].copy() for a in self.alphas}


def pca_basis(X, dim):
    """포스터 임베딩만으로 PCA 축을 계산 (평점 정보 미사용). 반환: (d, dim) 투영 행렬."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:dim].T.astype(np.float32)


def quantize(P, precision):
    """프로필 저장 정밀도 모사. int8 은 프로필마다 fp32 스케일 1개(4바이트)를 함께 저장."""
    if precision == "fp32":
        return P
    if precision == "fp16":
        return P.astype(np.float16).astype(np.float32)
    if precision == "int8":
        s = np.abs(P).max(1, keepdims=True) / 127.0
        return (np.round(P / s).clip(-127, 127) * s).astype(np.float32)
    raise ValueError(precision)


def profile_bytes(dim, precision):
    return dim * {"fp32": 4, "fp16": 2, "int8": 1}[precision] + (4 if precision == "int8" else 0)
