# Week 4 — 검색증강생성(RAG) 기초

문서 근거 기반 Q&A 어시스턴트를 기초부터 만들어봅니다: 일반 LLM만으로 충분하지 않은 이유, 임베딩과 유사도로 관련 텍스트를 찾는 방법, 벡터 데이터베이스로 그 검색을 확장하는 방법, 그리고 검색된 출처만으로 답변하는 방법을 다룹니다.

| Day | 주제 | 노트 |
|---|---|---|
| Day 1 | [왜 당신의 LLM은 회사 매뉴얼을 모를까](daily/Day1_llm-limits-and-rag-intro.md) | 환각, 지식 컷오프, 사적 데이터 공백; 오픈북 시험 비유; RAG 4단계 흐름 |
| Day 2 | [텍스트를 비교 가능한 숫자로 바꾸기](daily/Day2_embeddings-and-cosine-similarity.md) | 임베딩, 처음부터 만든 n-그램 토이 임베딩, 코사인 유사도 랭킹, 철자 vs. 의미의 한계 |
| Day 3 | [임베딩을 대규모로 저장하기: 벡터 데이터베이스](daily/Day3_vector-databases.md) | 무차별 대입 방식이 확장되지 않는 이유, 핵심 collection/add/query 연산, 거리 vs. 유사도, 순수 파이썬 폴백 |
| Day 4 | [문서를 청크로 나누고 검색된 내용만으로 답변하기](daily/Day4_chunking-and-grounded-answers.md) | 문서 전체를 보내지 않는 이유, 청크 크기/overlap 트레이드오프, 인용과 "모른다"를 포함한 `answer()` 패턴 |

4일치를 모두 담은 실행 가능한 노트북 1개는 [`concepts/4th_week_Concepts.ipynb`](concepts/4th_week_Concepts.ipynb)를 참고하세요. 짧고 독립적인 예제로 구성되어 있습니다.
