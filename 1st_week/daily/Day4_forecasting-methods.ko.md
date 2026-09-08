# Day 4 — 예측 기법과 평가

어제는 시리즈를 트렌드, 계절성, 잔차로 분해했다. 오늘은 그 구조를 이용해 다음에 무슨 일이 일어날지 실제로 예측하고, 그 예측이 얼마나 틀릴 가능성이 있는지를 정직하게 측정한다.

## 나이브에서 계절성까지: 방법의 사다리

네 가지 예측 방법이 있으며, 각각은 그 앞 방법의 엄격한 상위 집합이다 — 즉 더 나중 방법은 추가된 구성 요소를 끄면 수학적으로 이전 방법과 같아진다.

```mermaid
flowchart LR
    MA["이동평균<br/>평탄한 예측, 트렌드·계절 없음"] --> SES["단순 지수평활<br/>가중 최근 평균, 여전히 평탄"]
    SES --> HOLT["Holt 방법<br/>+ 트렌드 성분: 기울기 가능"]
    HOLT --> HW["Holt-Winters<br/>+ 계절 성분: 패턴 반복 가능"]
```

- **이동평균 예측** — 내일을 최근 N일의 단순 평균으로 예측한다. 트렌드나 계절성 개념이 전혀 없다. 성장하는 시리즈에서는 구조적으로 과소 예측한다(트렌드에 뒤처지도록 설계되어 있다). 반복되는 패턴도 완전히 무시한다.
- **단순 지수평활(SES)** — 최근 지점일수록 지수적으로 더 큰 가중치를 받는 가중 평균으로, 평활 파라미터 `alpha`(`0 < alpha <= 1`, 값이 클수록 최근 데이터에 더 큰 가중치)로 조절된다. 여전히 평탄한 직선 예측만 낼 수 있다 — 트렌드를 외삽하거나 계절 주기를 반복시킬 메커니즘이 없고, 현재 평활된 수준만 추적할 수 있다.
- **Holt 방법** — 트렌드의 기울기를 추적하는 두 번째 평활 성분을 추가해서, 예측이 평탄하게 유지되는 대신 위나 아래로 기울어질 수 있게 한다.
- **Holt-Winters** — 계절성을 위한 세 번째 평활 성분(가산 또는 승산, Day 3와 같은 구분)을 추가해서, 트렌드 위에 주간/월간/연간 패턴을 반복시키는 예측을 만든다.

Holt-Winters의 계절 성분을 끄면 수학적으로 Holt가 되고, Holt의 트렌드 성분까지 끄면 수학적으로 SES가 된다. 이는 단순한 설명을 위한 단순화가 아니라, 아래 나올 `trend=`와 `seasonal=` 인자를 통해 `statsmodels` API가 실제로 그렇게 구조화되어 있는 것이다.

## 시간 순서를 존중하는 분할

시계열에 무작위 train/test 분할을 절대 쓰지 마라. 분할 전에 행을 섞으면 컷오프 *이후*의 지점들이 학습 데이터로 새어 들어가서, 모델이 가까이 새어 들어온 지점들을 통해 사실상 "미래를 보게" 된다 — 이는 정확도 추정치를 살짝 편향시키는 수준이 아니라, 쓸 수 없는 모델을 훌륭해 보이게 만들 수 있다. 매끄러운 시리즈에서 시간적으로 인접한 지점들은 애초에 서로 매우 비슷하기 때문이다(Day 3의 자기상관 논의를 떠올려 보라 — 인접 지점 사이의 그 유사성이 바로 데이터 누수의 경로다). 항상 하나의 시간 기준점에서 잘라서 분할하라. 그 이전은 모두 train, 그 이후는 모두 test다.

```python
train, test = foot_traffic[:-60], foot_traffic[-60:]
# foot_traffic: Series, len=730 -> train: Series, len=670, test: Series, len=60 (마지막 60일을 홀드아웃)
```

## 홀드아웃 하나만으로는 충분하지 않을 수도 있다

시간 순서를 제대로 지킨 분할이라도 약점이 하나 있다. 정확히 하나의 60일짜리 구간에 대해서만 모델을 평가하기 때문에, 순전히 그 특정 구간에 우연히 무엇이 들어 있었는지(공휴일, 이상 한파 등)에 따라 점수가 유독 후하거나 유독 가혹하게 나올 수 있다. **워크포워드 검증(walk-forward validation, 롤링 오리진 평가라고도 한다)**은 시간 기준 분할을 여러 번 반복하면서 컷오프를 매번 앞으로 밀어내고, 폴드(fold) 전체에 걸쳐 점수를 평균 내어 이 문제를 해결한다.

```mermaid
flowchart LR
    F0["fold 0: 9월 1일까지 학습 -> 9월 2일~10월 1일 테스트"] --> F1["fold 1: 10월 1일까지 학습 -> 10월 2일~31일 테스트"]
    F1 --> F2["fold 2: 10월 31일까지 학습 -> 11월 1일~30일 테스트"]
    F2 --> F3["fold 3: 11월 30일까지 학습 -> 12월 1일~30일 테스트"]
```

모든 폴드가 여전히 시간 순서를 지킨다 — 학습 데이터는 항상 대응하는 테스트 구간이 시작되기 전에 끝난다 — 하지만 폴드마다 컷오프가 앞으로 이동하므로, 최종 점수는 우연히 운이 좋았거나(혹은 나빴던) 구간 하나가 아니라 시리즈의 여러 구간에 걸친 성능을 반영한다. 같은 시리즈에 대해 30일짜리 폴드 4개를 실행해서 Holt-Winters와 이동평균 베이스라인을 비교하면:

```python
def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))

horizon, n_folds = 30, 4
hw_scores, ma_scores = [], []
for fold in range(n_folds):
    cutoff = len(foot_traffic) - horizon * (n_folds - fold)
    train_fold = foot_traffic.iloc[:cutoff]
    test_fold = foot_traffic.iloc[cutoff:cutoff + horizon]

    hw = ExponentialSmoothing(train_fold, trend="add", seasonal="add", seasonal_periods=7).fit()
    hw_forecast_fold = hw.forecast(horizon)                                  # Series, len=30
    ma_forecast_fold = pd.Series([train_fold.iloc[-7:].mean()] * horizon, index=test_fold.index)

    hw_scores.append(mae(test_fold, hw_forecast_fold))
    ma_scores.append(mae(test_fold, ma_forecast_fold))
```

실제 폴드별 결과:

```
fold 0: HW_MAE=3.29  MA_MAE=11.34
fold 1: HW_MAE=3.11  MA_MAE=9.82
fold 2: HW_MAE=3.08  MA_MAE=9.97
fold 3: HW_MAE=3.55  MA_MAE=9.77

HW  mean MAE across folds: 3.26  std: 0.19
MA  mean MAE across folds: 10.22  std: 0.65
```

두 가지가 눈에 띈다. 첫째, 순위(Holt-Winters가 압도적으로 앞선다는 것)가 앞서 예제의 구간 하나뿐 아니라 모든 폴드에서 그대로 유지된다 — 바로 이 일관성이 단일 분할로는 얻을 수 없는 신뢰성을 워크포워드 결과에 부여한다. 둘째, 폴드 간 표준편차를 보라. Holt-Winters의 오차는 평균이 더 낮을 뿐 아니라 훨씬 더 *일관적이다*(표준편차 0.19 대 0.65) — 이동평균 베이스라인의 정확도는 우연히 테스트받은 30일짜리 구간이 주간 주기의 어느 부분에 해당하는지 전혀 예측할 방법이 없기 때문에 눈에 띄게 더 크게 흔들린다.

## 예제로 확인하기: 네 방법을 모두 피팅하고 점수 매기기

Day 3와 동일한 2년치 합성 유동인구 시리즈(트렌드 + 주간 계절성 + 노이즈)를 사용해, 마지막 60일을 홀드아웃으로 남기고 네 가지 방법을 모두 피팅했다 — `statsmodels` 0.14.6으로 처음부터 끝까지 직접 실행했다.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def moving_average_forecast(train, horizon, window=7):
    last_avg = train.iloc[-window:].mean()
    return pd.Series([last_avg] * horizon, index=test.index)   # 마지막 윈도우 평균값의 평탄선

ma_forecast = moving_average_forecast(train, len(test))

ses_model = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
ses_forecast = ses_model.forecast(len(test))

holt_model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
holt_forecast = holt_model.forecast(len(test))

hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit()
hw_forecast = hw_model.forecast(len(test))
# hw_forecast: Series, len=60 -- 홀드아웃 일자마다 예측값 하나씩
```

실제 홀드아웃 값과 비교해 MAE, MAPE, RMSE(아래에서 정의)로 점수를 매기면:

```
Method            MAE    MAPE(%)   RMSE
MovingAvg          9.80    6.34    11.46
SES               10.17    6.66    11.87
Holt              11.29    7.53    13.41
Holt-Winters       3.32    2.12     4.17
```

Holt-Winters는 모든 지표에서 다른 모든 방법보다 대략 **3배** 더 좋은 결과를 낸다 — 미미한 승리가 아니다. 이는 여기서 예상된 결과이며, *왜* 그런지 이해할 가치가 있다. 이 합성 시리즈는 정확히 주기-7인 강한 주간 파동을 가지고 있고, 이는 정확히 Holt-Winters만이 모델링하고 있는 구조다. 나머지 세 방법은 그 예측 가능한 주간 진동 전체를 모델링되지 않은 오차로 접어 넣을 수밖에 없고, 그래서 오차가 일관되게 3배 더 크다. Holt가 단순 SES보다 오히려 *더 나쁜* 결과(RMSE 13.41 vs 11.87)를 낸 것도 주목할 만하다 — 트렌드 성분을 추가한다고 자동으로 도움이 되는 것은 아니다. 훈련 윈도우의 *국소적* 트렌드는 60일짜리 예측 구간에서 살짝 잘못된 방향으로 외삽될 수 있는데, 더 평탄한 SES 예측은 우연히 그 함정을 피해간 것이다. 모델 복잡도가 늘어난다고 자동으로 정확도가 올라가는 것은 아니다 — 추가된 성분이 실제로 데이터에 있는 구조와 맞아떨어질 때만 그 대가를 받는다. 바로 이 때문에 Day 3의 분해 단계(여기 정말로 깨끗한 주기-7 계절성이 있는지 확인하는 것)가 이 단계보다 나중이 아니라 먼저 와야 한다.

## 점 하나가 아니라 범위

예측값 하나만 제시하는 것은 항상 실제보다 더 큰 정밀도를 암시한다. Holt-Winters는 잔차 오차 분포가 추정된 통계 모델로 피팅되므로, 피팅된 잔차와 일치하는 노이즈를 반복적으로 샘플링해 그럴듯한 미래 경로를 여러 개 **시뮬레이션**하고, 한 줄의 숫자 대신 백분위 범위를 보고할 수 있다.

```python
simulations = hw_model.simulate(len(test), repetitions=200, error="add", random_state=7)
# simulations: DataFrame, shape (60, 200) -- 예측 60일 x 시뮬레이션 경로 200개

lower = simulations.quantile(0.05, axis=1)   # Series, len=60 -- 일자별 5번째 백분위
upper = simulations.quantile(0.95, axis=1)   # Series, len=60 -- 일자별 95번째 백분위
```

이를 실행하고 *실제* 홀드아웃 값이 그 90% 밴드 안에 실제로 몇 번 들어왔는지 확인해 보면, 60개 실제 테스트 지점 중 **83.3%**가 그 안에 들어왔다. 이는 명목상의 90%보다 다소 낮은 수치다 — 시뮬레이션된 구간도 그것을 만들어낸 모델만큼만 정확할 수 있다는 유용하고 정직한 사실을 보여준다. 여기서는 그 불일치가 위에서 언급한 Holt 대 SES의 트렌드 외삽 흔들림과 같은 원인에서 비롯되며, 이는 시뮬레이션된 경로들의 중심 경향을 실제 시리즈가 실제로 향한 지점에서 살짝 벗어나게 만든다. 예측 구간은 "여기가 그럴듯한 범위이고, 실제로는 대략 이 정도 비율로 그 안에 들어와야 한다"는 것을 전달한다 — 이는 보정이 완벽하지 않더라도, 확신에 찬 것처럼 보이는 한 줄의 숫자보다 근본적으로 더 정직한 결과물이다. 모델의 불확실성을 숨기는 대신 눈에 보이게 만들어주기 때문이다.

## 정상성(Stationarity)과 차분

ARIMA를 비롯한 많은 고전적 예측 기법은 시리즈가 **정상적(stationary)**이라고 가정한다. 즉 통계적 성질(평균, 분산, 자기상관 구조)이 시간에 따라 변하지 않는다는 것이다. 명확한 트렌드가 있는 시리즈는 정의상 정상적이지 않다. 시작 지점과 끝 지점의 평균이 다르기 때문이다.

**증강 디키-풀러(ADF) 검정**은 이를 공식적으로 확인한다. 귀무가설은 "이 시리즈는 정상적이지 않다"이므로, p-값이 낮으면(관례적으로 < 0.05) 그 귀무가설을 기각하고 시리즈가 정상적일 가능성이 크다고 결론 내릴 수 있다.

트렌드가 있는 유동인구 시리즈와 그 1차 차분에 대해 실행하면:

```python
from statsmodels.tsa.stattools import adfuller

stat, p_value, *_ = adfuller(foot_traffic)
# 원본 시리즈: stat=-0.568, p=0.8780 -- 비정상성을 기각할 수 없음 (예상대로: 실제 트렌드가 있음)

diffed = foot_traffic.diff().dropna()
# foot_traffic: Series, len=730 -> diffed: Series, len=729 (한 지점 손실: 차분에는 이전 값이 필요)

stat_d, p_value_d, *_ = adfuller(diffed)
# 차분된 시리즈: stat=-10.298, p≈0.0000003 -- 강하게 정상적
```

**차분(differencing)** — 각 값을 이전 값과의 차이로 대체하는 것(`foot_traffic.diff()`) — 이 트렌드를 완전히 제거해서, 시리즈를 명백히 비정상적인 상태(p=0.878)에서 강하게 정상적인 상태(p가 사실상 0)로 뒤바꿨다. 이는 교과서적인 주장을 그냥 믿는 게 아니라 실제 숫자로 확인한 것이다. 선형 트렌드는 정확히 1차 차분 한 번이면 제거하도록 설계된 종류의 비정상성이다.

## 방법들에 점수 매기기

각각 살짝 다른 질문에 답하는 세 가지 표준 오차 지표가 있다.

- **MAE**(평균 절대 오차) — 원래 단위 기준으로 오차의 평균 크기. 비전문가 이해관계자에게 설명하기 쉽다("하루 평균 방문객 약 3명 정도 차이가 납니다").
- **MAPE**(평균 절대 백분율 오차) — 같은 개념을 백분율로 스케일링해서, 스케일이 크게 다른 시리즈(하루 방문객 수 대 월 매출 달러)끼리도 비교할 수 있게 해준다 — 하지만 실제 값으로 나누기 때문에 0 근처에서는 불안정하거나 무의미해질 수 있다.
- **RMSE**(평균 제곱근 오차) — MAE와 비슷하지만, 평균을 내기 전에 오차를 제곱하기 때문에 큰 오차가 작은 오차보다 불균형하게 더 크게 벌점을 받는다. 몇 번의 큰 실수가 여러 번의 작은 실수보다 더 중요한 경우(예: 재고 부족)에는 RMSE를, 크기와 상관없이 모든 오차를 동등하게 취급하고 싶다면 MAE를 선호하라.

```python
import numpy as np

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def mape(y, yhat): return float(np.mean(np.abs((y - yhat) / y)) * 100)
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
```

## 흔한 함정

- **시간 순서가 있는 데이터에 무작위 train/test 분할을 쓰기.** 위에서 다뤘지만, 백테스트 결과를 비현실적으로 좋게 만드는 가장 흔한 방법이므로 별도의 함정으로 다시 강조할 가치가 있다.
- **0 근처에서 MAPE를 신뢰하기.** 정당하게 거의 0에 가까운 값까지 떨어지는 시리즈(예: B2B 제품의 주말 매출)는 절대 오차가 작고 특별할 게 없어도, 순전히 분모 때문에 MAPE가 폭발하거나 정의되지 않을 수 있다.
- **구조가 실제로 있는지 확인하지 않고 모델 복잡도를 추가하기.** 위 Holt vs SES 결과가 직접 보여주듯, 더 화려한 모델이 자동으로 더 나은 모델은 아니다. 추가된 복잡도가 제 값을 했는지 신뢰하기 전에, 항상 같은 홀드아웃 윈도우에서 더 단순한 방법들과 비교하라.
- **불확실성 밴드 없이 점 예측만 보고하기.** 특히 비즈니스 의사결정(인력 배치, 재고)에 들어가는 무언가라면, 숫자 하나만 제시하는 것은 위의 시뮬레이션 기반 범위가 특별히 막으려는 그 거짓된 확신을 불러일으킨다.

## 요약

데이터에 실제로 있는 구조를 포착하는 가장 단순한 방법을 골라라 — Holt-Winters가 여기서 3배나 이긴 것은 데이터에 진짜로 깨끗한 주간 계절성이 있기 때문일 뿐이다. 진짜 계절 구조가 없는 시리즈에서는 그 추가 장치가 순수한 오버헤드가 되고, 위에서 Holt가 SES보다 나빴던 것처럼 오히려 더 단순한 방법보다 성능이 떨어질 수도 있다. 항상 무작위 분할이 아니라 시간 순서를 지키는 홀드아웃으로 평가하고, 단일한 "최선의 추측" 숫자보다 시뮬레이션된 예측 구간을 보고하는 쪽을 선호하라 — 명목상 90%인 밴드에 대한 83%의 실측 적중률 자체가 이 모델을 얼마나 신뢰해야 할지에 대한 유용한 정보이며, 단일 점 예측이었다면 완전히 감춰졌을 정보다.
