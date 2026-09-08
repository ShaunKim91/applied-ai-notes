# Week 3 — 멀티모달 문서 & 이미지 AI

직접 비전 모델을 학습시키는 대신 사전학습된 멀티모달 모델을 활용해 사진, PDF, HTML 페이지를 구조화되고 신뢰할 수 있는 데이터로 바꾸는 방법을 다룹니다.

| Day | 주제 | 노트 |
|---|---|---|
| Day 1 | [학습 없이 보기: 사전학습된 멀티모달 모델](daily/Day1_multimodal-vision-intro.md) | CNN 개념 정리, CNN 학습보다 API 호출이 나은 이유, mock-first 래퍼 패턴 |
| Day 2 | [사진을 구조화된 JSON으로](daily/Day2_image-to-structured-json.md) | 프롬프트 기반 JSON 추출, `safe_json()` 파싱, 객체 탐지와의 비교, PII 처리 |
| Day 3 | [PDF를 읽고 지어내지 않고 요약하기](daily/Day3_pdf-parsing-and-summarization.md) | `pypdf` 텍스트 추출, 스캔 PDF 폴백, 청크 단위 map-reduce 요약, 숫자 근거 검증 |
| Day 4 | [웹 페이지를 CSV 리포트로 바꾸기](daily/Day4_html-table-parsing-and-reporting.md) | BeautifulSoup 파싱, 데이터프레임 변환, 스크래핑 에티켓, 배치 오류 처리 |

4일치를 모두 담은 실행 가능한 노트북 1개는 [`concepts/3rd_week_Concepts.ipynb`](concepts/3rd_week_Concepts.ipynb)를 참고하세요. 짧고 독립적인 예제로 구성되어 있습니다.
