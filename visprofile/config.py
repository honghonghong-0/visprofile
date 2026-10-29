"""경로와 실험 설정값을 한 곳에서 관리한다.

데이터 루트는 환경변수 VISPROFILE_ROOT 로 지정한다. 하위 구조:
    {root}/data/movielens/ml-20m/ratings.csv, movies.csv
    {root}/data/movielens/psm/ml-20m-psm/posters/{movieId}.jpg
    {root}/data/features/   (임베딩 저장 위치)
    {root}/results/         (결과 저장 위치)
"""
import os
from dataclasses import dataclass, field

import pandas as pd

DAY = 86400  # 초 단위 타임스탬프 -> 일 단위 변환
TOP_K = 10


@dataclass
class Paths:
    root: str = field(default_factory=lambda: os.environ.get("VISPROFILE_ROOT", "."))

    @property
    def ratings(self):
        return f"{self.root}/data/movielens/ml-20m/ratings.csv"

    @property
    def movies(self):
        return f"{self.root}/data/movielens/ml-20m/movies.csv"

    @property
    def posters(self):
        return f"{self.root}/data/movielens/psm/ml-20m-psm/posters"

    @property
    def features(self):
        return f"{self.root}/data/features"

    @property
    def results(self):
        os.makedirs(f"{self.root}/results", exist_ok=True)
        return f"{self.root}/results"

    def embedding(self, encoder):
        return f"{self.features}/clip_{encoder}.npy"

    @property
    def movie_ids(self):
        return f"{self.features}/poster_movie_ids.npy"


@dataclass
class Config:
    encoder: str = "ViT-L-14_openai"
    like_threshold: float = 4.0    # 평점 4 이상 = 선호
    drop_first: int = 15           # 가입 초기 평점 제거 개수
    min_ratings: int = 20          # 사용자 필터: 최소 평점 수
    min_days: int = 5              # 사용자 필터: 최소 평가 일수
    min_pre_liked: int = 5         # 기준 시점 이전 최소 선호작 수
    min_eval_days: int = 2         # 기준 시점 이후 최소 평가일 수
    alphas: tuple = (0.05, 0.1, 0.2, 0.3, 0.5)   # EMA 후보 (검증 구간에서 선택)
    hybrid_weights: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)
    recent_window_days: int = 365  # 최근 인기도 / 전역 최근 프로필 창
    val_cut: str = "2008-01-01"
    test_cut: str = "2009-01-01"
    min_free_gpu_mib: int = 15000  # 공용 GPU 보호용 임계치

    @staticmethod
    def to_day(date):
        return int(pd.Timestamp(date).timestamp() // DAY)
