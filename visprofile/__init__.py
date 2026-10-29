"""visprofile: 단말 기반 연속 갱신 영상 콘텐츠 시각 선호 프로필.

모듈 구성
  config        경로·실험 설정
  data          평점/임베딩 적재, 사용자 필터
  encoder       포스터 -> CLIP 임베딩 (정사각형 여백 처리)
  user_profile  단말 프로필 상태(누적 평균·EMA) 갱신, PCA·양자화
  stream        일 단위 연속 평가 스트림 구성
  baselines     인기도·전역 최근 프로필 (서버 측 신호)
  evaluate      후보 점수화와 Recall/NDCG 계산
  stats         사용자 단위 Wilcoxon 검정, Holm 보정
"""
__version__ = "1.0.0"
