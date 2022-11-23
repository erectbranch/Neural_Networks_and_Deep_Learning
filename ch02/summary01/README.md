# 얕은 신경망을 이용한 기계 학습

## 2.1 소개

대체로 전통적인 기계 학습은 optimization(최적화)와 backpropagation(경사 하강법)을 이용해서 parameterized model을 학습한다. 이런 model의 예시로 linear regression, SVM, logistic regression, dimention reduction, matrix decomposition(행렬 인수분해) 등이 있다.

![퍼셉트론의 여러 변형](images/perceptron_machine_learning.png)

이번 장은 기계 학습의 아주 다양한 optimization 중심 방법들은 layer가 1~2개뿐인 아주 단순한 신경망 구조로 흉내 낼 수 있음을 보일 것이다.

---

## 2.2 binary classification(이진 분류) model을 위한 신경망 구조

이번 절 전반은 input node가 d개, output node가 1개인 퍼셉트론을 사용한다.

- $\bar{W} = (w_1, ... , w_d)$ : weight

- bias 항은 굳이 두지 않고, 값이 1인 가짜 input node를 하나 추가해서, 그 계수를 bias 항처럼 쓸 것이다.

### 2.2.1 퍼셉트론 다시 보기

예측값은 다음과 같이 쓸 수 있다.

$$ \hat{y_i} = sgn(\overline{W} \cdot \overline{X_i}) $$

- $(\overline{X_i}, y_i)$ : 하나의 훈련 견본

퍼셉트론 갱신 공식은 다음과 같았다.(regularization이 적용된)

$$ \overline{W} \Leftarrow \overline{W}(1-\alpha\lambda) + \alpha(y_i - \hat{y_i}) \overline{X_i} $$

backpropagation의 갱신이 이처럼 오차에 비례하는 경우, $(y_i - \hat{y_i})^2$ 같은 제곱 손실함수를 사용하는 것이 자연스럽다. 그런데 지금 출력은 이산값이기 때문에, loss function의 값마저 이산적이게 된다. 지점마다 계단 모양을 그리기 때문에 미분할 수 없다.

그래서 1.2.1.1절에서 정리했듯 미분 가능한 smoothed surrogate loss function(평활화된 대리 함수)를 대신 사용한다고 정리했다. 이때 오분류된 훈련 견본 ( $y_i \hat{y_i}$ < 0 )에서만 weight 갱신이 일어났다. 

갱신 공식을 indicator function(지시함수: 인수로 주어진 조건이 성립하면 1, 그렇지 않으면 0) $I(\cdot) \in \lbrace 0, 1 \rbrace$ 을 사용해서 다음과 같이 표현할 수 있다.

$$ \overline{W} \Leftarrow \overline{W}(1-\alpha\lambda) + \alpha y_i \overline{X_i}[I(y_i \hat{y_i} < 0)] $$

- 오분류된 견본일 경우 $(y_i - \hat{y_i}) / 2$ 가 예측값이므로, 위와 같이 $(y_i - \hat{y_i})$ 오차 부분을 수정했다.(계수 2는 learning rate로 흡수)

i번째 훈련 견본에 대한 loss function은 다음과 같았다. 또한 이 loss function을 **perceptron criterion**(퍼셉트론 판정기준)이라고 불렀다.

$$ L_i = \max \lbrace 0, -y_i(\overline{W} \cdot \overline{X}_i) \rbrace $$

---

### 2.2.2 least-squares regression(최소제곱 회귀)

least-squares regression에서 train data는 n개의 서로 다른 훈련 견본 $(\overline{X_1}, y_1)...(\overline{X_n}, y_n)$ 으로 이루어진다. 

- 여기서 각 $\overline{X_i}$ data point는 d차원 표현이고, $y_i$ 는 real(실수) target(목푯값)이다. 실수이기 때문에 regression 문제가 된다.

i번째 훈련 견본에 대한 loss function은 다음과 같다.

$$ L_i = e_i^2 = (y_i - \hat{y_i})^2 $$

backpropagation 갱신 공식은 다음과 같다.

$$ \overline{W} \Leftarrow \overline{W} + \alpha e_i \overline{X} $$

다음과 같이 표현할 수도 있다.

$$ \overline{W} \Leftarrow \overline{W} + \alpha (y_i - \hat{y_i}) \overline{X} $$

regularization을 적용할 수도 있다.

$$ \overline{W} \Leftarrow \overline{W}(1 - \alpha \cdot \lambda) + \alpha (y_i - \hat{y_i}) \overline{X} $$

이 갱신 공식은 2.2.1의 perceptron criterion과 매우 비슷하다. 그러나 두 갱신 공식은 완전히 같은 것은 아니다.( $\hat{y_i}$ 계산 방식이 다르기 때문)

그렇다면 이진 target에 적용하면 어떨까? 이 경우는 least-squares classification(최소제곱 분류) 문제가 된다. 이 경우 perceptron creterion과 겉보기에 동일한데, 퍼셉트론 알고리즘과 결과가 같지는 않다.

그 이유는 least-squares classification의 '실숫값' 훈련 오차 $(y_i - \hat{y_i})$ 와, 퍼셉트론 '정수' 오차 $(y_i - \hat{y_i})$ 의 계산 방식이 완전히 다르기 때문이다.

> 이런 least-squares classification(최소제곱 분류)를 Widrow-Hoff learning(위드로-호프 학습)이라고 한다.

---

### 2.2.2.1 Widrow-Hoff learning(위드로-호프 학습)

기존의 least-squares regression을 binary target에 적용하고자 하는 시도에서 탄생했다. 

Widrow-Hoff learning은 미지의 시험 견본의 실수 예측값을 sign function을 이용해서 binary target으로 변환하긴 하지만, <U>훈련 견본과의 오차는 실수 예측값을 직접 사용해서 계산한다.</U> 

> 퍼셉트론의 경우 오차는 항상 {-2, +2}에 속하지만, Widrow-Hoff learning의 경우 $\hat{y_i}$ 가 sign function 없이 $\overline{W} \cdot \overline{X_i}$ 로 결정되기 때문에 오차가 임의의 실숫값이다.

이 때문에 positive 부류에 속하는 data point가 $\overline{W} \cdot \overline{X_i} > 1$ 인 경우에서 차이를 보일 수 있다. 퍼셉트론은 벌점이 가해지지 않겠지만, Widrow-Hoff에서는 실수 예측값 오차이기 때문에 벌점이 가해지게 된다.

<U>성과가 너무 좋은 point에도 부당하게 벌점이 가해지는 것</U>이 이 Widrow-Hoff learning의 단점이다.

이 방법의 loss function은 다음과 같다.

$$ L_i = (y_i - \hat{y_i})^2 = y_i^2(y_i - \hat{y_i})^2 = (1-\hat{y_i}y_i)^2 $$

- $y_i^2 = 1$ 이므로 loss function에 곱하는 것으로 모양을 바꾼 것이다.

> 위 loss function을 '너무 좋은 성과'에 벌점을 부여하지 않도록 수정하는 한 방법을 적용하면 SVM의 loss function이 된다.

---