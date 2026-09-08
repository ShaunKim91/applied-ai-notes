# Week 3 — 멀티모달 문서 & 이미지 AI

직접 비전 모델을 학습시키는 대신 사전학습된 멀티모달 모델을 활용해 사진, PDF, HTML 페이지를 구조화되고 신뢰할 수 있는 데이터로 바꾸는 방법을 다룹니다.

| Day | 주제 | 노트 |
|---|---|---|
| D1 | [학습 없이 보기: 사전학습된 멀티모달 모델](D1/daily/Day1_multimodal-vision-intro.md) | CNN 개념 정리, CNN 학습보다 API 호출이 나은 이유, mock-first 래퍼 패턴 |
| D2 | [사진을 구조화된 JSON으로](D2/daily/Day2_image-to-structured-json.md) | 프롬프트 기반 JSON 추출, `safe_json()` 파싱, 객체 탐지와의 비교, PII 처리 |
| D3 | [PDF를 읽고 지어내지 않고 요약하기](D3/daily/Day3_pdf-parsing-and-summarization.md) | `pypdf` 텍스트 추출, 스캔 PDF 폴백, 청크 단위 map-reduce 요약, 숫자 근거 검증 |
| D4 | [웹 페이지를 CSV 리포트로 바꾸기](D4/daily/Day4_html-table-parsing-and-reporting.md) | BeautifulSoup 파싱, 데이터프레임 변환, 스크래핑 에티켓, 배치 오류 처리 |

4일치를 모두 담은 실행 가능한 노트북 1개는 [`concepts/Week3_Concepts.ipynb`](concepts/Week3_Concepts.ipynb)를 참고하세요. 짧고 독립적인 예제로 구성되어 있습니다.
