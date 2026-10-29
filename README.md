# visprofile: 단말 기반 연속 갱신 영상 콘텐츠 시각 선호 프로필

사용자가 선호한 영화 포스터의 CLIP 임베딩을 고정 크기 벡터(시각 선호 프로필)로 요약하고, 새 상호작용이 발생할 때마다 과거 기록을 다시 읽지 않고 O(d)로 갱신하는 단말 지향 사용자 프로파일링 소프트웨어이다. MovieLens-20M에서 시간 순서를 보존한 일 단위 연속 평가, 프로필 압축(PCA·양자화) 분석, 통계 검정, 논문 그림 생성을 포함한다.

- 논문: 「단말 기반 연속 갱신을 위한 영상 콘텐츠 시각 선호 프로필」, 2026 한국방송·미디어공학회 추계학술대회
- 저작자: 홍서현, 황찬빈, 강정완 (성균관대학교)

## 구성

```
visprofile/              핵심 패키지
  config.py              경로·실험 설정 (필터 기준, 기준 시점, EMA 후보 등)
  data.py                평점·임베딩 적재, 포스터 임베딩 중심화, 사용자 필터
  encoder.py             포스터 정사각형 여백 처리 + CLIP 임베딩 추출
  user_profile.py        단말 프로필 상태(누적 평균·EMA) 갱신, PCA 축, 양자화
  stream.py              일 단위 연속 평가 스트림(예측 → 관측 → 갱신) 구성
  baselines.py           누적/최근 인기도, 전역 최근 프로필 (서버 측 신호)
  evaluate.py            후보 점수화, Recall@10·NDCG@10·신규 영화 Recall@10, GPU 선택
  stats.py               사용자 단위 Wilcoxon 부호순위 검정, Holm 보정
scripts/                 실행 스크립트 (번호 순서대로 실행)
  00_data_check.py       포스터 커버리지, 기준 시점 T 후보별 평가 가능 사용자 수
  01_extract_embeddings.py   포스터 CLIP 임베딩 추출
  02_embedding_diagnostics.py 중심화 전/후 유사도 분포 진단
  03_stream_eval.py      RQ1·RQ2 연속 평가 (검증·테스트·순서 대조군)
  04_compression.py      RQ3 프로필 압축 분석
  05_significance.py     사전 지정 비교의 유의성 검정
  06_figures.py          논문 그림 1–3 생성
```

## 설치

```bash
conda create -p ./env -c conda-forge python=3.11
./env/bin/pip install -r requirements.txt
# GPU 드라이버가 CUDA 12.8 인 경우
./env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 데이터 준비

`VISPROFILE_ROOT` 아래에 다음 구조로 둔다.

```
data/movielens/ml-20m/            GroupLens MovieLens-20M (ratings.csv, movies.csv)
data/movielens/psm/ml-20m-psm/posters/   {movieId}.jpg  (Zenodo doi:10.5281/zenodo.14571726)
```

## 실행

```bash
export VISPROFILE_ROOT=/path/to/root
python scripts/00_data_check.py
python scripts/01_extract_embeddings.py --encoder ViT-L-14_openai
python scripts/02_embedding_diagnostics.py
python scripts/03_stream_eval.py val test test_shuffled test_shuffled_all
python scripts/04_compression.py
python scripts/05_significance.py
python scripts/06_figures.py
```

결과는 `$VISPROFILE_ROOT/results/` 에 저장된다. GPU 는 여유 메모리가 가장 큰 장치를 자동 선택하며, 여유 메모리가 `Config.min_free_gpu_mib` 미만이면 실행을 중단한다(공용 서버 보호).

## 평가 설정 요약

| 항목 | 값 |
|---|---|
| 선호 정의 | 평점 4 이상 |
| 사용자 필터 | 첫 15개 평점 제거 후 20개 이상, 5일 이상 |
| 기준 시점 | 검증 2008-01-01 ~ 2009-01-01, 테스트 2009-01-01 이후 |
| 갱신 단위 | 하루 (같은 날 선호작 평균으로 1회 갱신) |
| 후보 | 예측일 이전에 평가된 적 있는 영화 − 사용자가 이미 평가한 영화 |
| 지표 | 사용자별 평균 Recall@10, NDCG@10 |
