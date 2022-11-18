## 1.4 신경망 훈련의 실질적인 문제점들

## 1.4.1 overfitting(과대적합)

신경망 훈련에서 가장 중요한 문제점으로는 overfitting(과대적합)이 꼽힌다. 한 model을 train data set을 이용해서 훈련시킨 뒤, 모형이 label을 완벽하게 예측했다고 해도, 미지의 test data에 대해 좋은 예측 성과를 내리라는 보장이 없다는 것이다.

> train data에 너무 완벽하게 맞춰지면서 training error가 0이 되면, 반대로 실제 데이터에는 제대로 된 예측을 할 수 없을 것이다. 

> 학습에 사용되지 않은 데이터를 활용하여 평가하는 과정을 test를 한다고 하고, 이를 **Generalization**(일반화)라고 표현한다. train data와 test data는 서로 교점이 되는 데이터들이 없기 때문에, test를 Generalization이라고 하고 test error를 generalization error라고 표현하는 것이다.

train data를 늘리면 model의 일반화 성능이 개선되는 반면, model의 복잡도를 늘리면 일반화 성능이 줄어든다. 그렇다고 모형이 너무 단순하면 제대로 feature를 파악하지 못할 위험이 있다.

> 일반적인 법칙으로 전체 훈련점 개수가, 신경망 매개변수 개수의 적어도 2~3배는 되어야 한다.

> 일반적으로, 매개변수가 많은 model을 가리켜 '**capacity**(수용력)이 높다'라고 표현한다. 즉, high capacity model가 잘 일반화되기 위해서는 많은 훈련 자료가 필요하다.

보통 기계학습에서 overfitting을 bias와 variance의 trade-off 관점으로 볼 때가 많다. 모형의 복잡도를 결정할 때는 최적의 지점을 세심하게 선택할 필요가 있다.

> 신경망이 거의 모든 종류의 함수를 흉내낼 수 있을 정도로 강력했지만, 그동안 인기가 없었던 이유가 바로 이런 단점 때문이다. 지금은 가용할 수 있는 data가 많아지면서 기존 기계 학습에 비해 신경망의 장점이 매우 두드러진 것이다.

이제 overfitting의 영향을 완화하는 몇 가지 설계 방법을 소개할 것이다.

---

## 1.4.1.1 regularization(정칙화)

> [Regularization (Weight Decay)](https://deepapple.tistory.com/6)

매개변수가 많을수록 overfitting이 일어나므로, 이를 방지하기 위해 모형에서 (0을 제외한) <U>매개변수의 개수를 줄이는 방향</U>으로 접근할 수 있다.

또한 <U>매개변수의 절댓값의 크기를 줄여도 overfitting이 완화</U>되는 경향이 있는데, 그렇다고 매개변수 값 자체를 직접적으로 제한하는 것은 어렵다. 그래서 loss function에 penalty 항 $ \lambda || \overline{W} ||^p $ 를 도입하는 좀 더 온건한 접근 방식이 쓰인다. 

> 데이터가 단순하고 모델이 복잡하면, training을 하면서 작은 값이었던 weight들의 값이 점점 증가한다. weight가 커질수록 train data가 모델에 주는 영향력이 커지고, 결국 모델이 train data에 딱 맞춰지게 된다. 이것을 local noise의 영향을 크게 받아서, outlier들에 모델이 맞춰지는 현상이라고 표현한다.

이때 $p$ 는 흔히 2로 설정하는데, 이는 **L2 regularization**(Tikhonov regularization, 티호노프 정칙화)에 해당한다. L2 regularization에서는 각 매개변수(에 정칙화 매개변수 $\lambda >0 $ 을 곱한 값)을 제곱한 결과를 loss function에 더해준다.

---

### <span style='background-color: #393E46; color: #F7F7F7'>&nbsp;&nbsp;&nbsp;🔒 정의: L1, L2 Norm/Regularization&nbsp;&nbsp;&nbsp;</span>

> [L1 Regularization, L2 Regularization](https://light-tree.tistory.com/125)

1. **Norm**

**Norm**은 벡터의 크기(혹은 길이)를 측정하는 방법(혹은 함수)를 뜻한다. 

$$ ||x||_{p} := \left( \sum_{i=1}^{n}{|x_i|}^p \right)^{1/p} $$

- L1 Norm

$$ d_{1}(p,q) = {||p-q||}_{1} = \sum_{i=1}^{n}{|p_i - q_i|} $$

- $ p = (p_1, p_2, ..., p_n), \quad q = (q_1, q_2, ..., q_n) $

쉽게 말하면 두 벡터 $ p, q $ 의 각 원소들 간 차이의 절댓값들을 구한 뒤 모두 더한 값이다. 

- L2 Norm

$$ {||x||}_2 := \sqrt{{x_{1}}^{2} + ... + {x_{n}}^2} $$

- $ p = (p_1, p_2, ..., p_n), \quad q = (0, 0, ..., 0) $

- 공식을 간편하게 작성하기 위해 q를 원점으로 뒀을 뿐, 실제로 값이 있다면 차이를 대입하면 된다.

<br/>

2. **Regularization**

- L1 Regularization

기존의 cost function(loss function의 평균)에 $ \lambda |w| $ 를 더한 값을 cost function으로 쓰게 된다.

예를 들어 다음과 같은 수식이 있다고 가정하자.

$$ H(X) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 $$

여기에 L1 Regularization을 적용했을 때 $ w_3 $ 의 값이 0이 되었다면, 이 말은 $ x_3 $ 의 feature이 사실 model의 결과에 별 영향을 주지 못한다는 의미이다.

> 대부분의 구현에서 L2 Regularization을 선호하지만, L1 Regularization도 나름의 용도가 있다. input에서 나가는 edge의 $ w_i $ 의 값이 0이면 해당 input은 최종 output에 아무런 영향을 미치지 않게 된다. 즉, 그런 input이 생략되면서(dropped) 일종의 feature 선택기로 작용한다. 따라서 L1 Regularization을 적용하면서 feature의 영향을 파악해 볼 수 있다.

- L2 Regularization

벡터의 크기를 의미하는 Norm을 이용해서, 가중치 벡터 크기만큼의 penalty를 부여하는 개념이 바로 regularization이다.

> 이런 penalty 부여를 갱신 도중 일종의 'weight decay(가중치 감쇠)를 적용'하는 것으로 봐도 무방하다.

weight decay는 기존의 cost function에 항을 추가해서 큰 값을 가지는 weight에 penalty를 부여하는 것이다. 이때 L2 norm 항이 weight 패러미터의 크기를 나타내기 때문에, 현재 weight의 크기에 비례해서 더해주는 값이 커진다.

$$ E(w) = E_0(w) + {1 \over 2}\lambda\sum_{i}{w_{i}^2} $$

- $E_{0}(w)$ : 기존 cost function

> 앞에 1/2가 붙는 것은 미분의 편의성을 위해서이며, 아예 안 쓰거나 1/N으로 표기하는 경우도 많다. 

w를 행렬로 다시 쓰면 다음과 같이 표현할 수 있다.

$$ E(W) = MSE_{train} + {1 \over 2}\lambda W^{\mathsf{T}}W $$

그렇다면 왜 이 값을 더해주는 것이 penalty로 작용할까? 그 이유는 gradient descent를 생각해 보면 알 수 있다.

gradient descent는 error의 최솟값을 찾아가는 과정이다. 그런데 error 값에 일정 비율 '가중치 크기'에 해당하는 값을 더한다면, 가중치의 절대적인 크기가 클수록 error가 커지고 반영은 결과적으로 덜 이루어지게 된다.

---

아래는 정칙화를 적용한 갱신 공식이다.

$$ \overline{W} \Leftarrow \overline{W}(1-a\lambda) + \alpha \sum_{\overline{X} \in S}{E(\overline{X})\overline{X}} $$

- $E(\overline{X})$ : 오차 $ (y - \hat{y})$ (모형마다 고유의 오차함수가 들어간다.)

- $S$ : mini batch(미니배치)

- $ \alpha $ : learning rate를 조정하는 매개변수

- $ \lambda $ : 정칙화의 정도를 조절하는 정칙화 매개변수. 0이면 원래의 loss function이 되고, 0보다 크면 기울기 변화가 너무 커지지 않도록 감쇠한다.(대체로 0.01 ~0.00001)

> 생물학적으로 비유하면 뇌에서 덜 중요한, 즉 noise 패턴이 제거되면서 발생하는 '점진적 망각'이라고 볼 수 있다.

![regularization 적용 전후 비교](images/regularization_before_after.png)

아래 그림은 하이퍼패러미터 $ \lambda $ 값에 따른 multinomial regression 결과이다.

![lambda 값에 따른 multinomial regression](images/regularization_lambda.png)

### 정칙화의 결점

[Regularization in Neural Network](https://cedar.buffalo.edu/~srihari/CSE574/Chap5/Chap5.5-Regularization.pdf)

1. **Invariance to Transformation**

하지만 weight decay를 적용하기 전에 몇몇 <U>scaling 과정을 적용했다면</U> 지금의 방법을 적용할 수 없게 된다. 예시로 2 layer로 구성된 다층 퍼셉트론에 linear transform을 적용하면 어떻게 weight decay를 쓸 수 없게 되는지 보자.

2 layer로 구성된 다층 퍼셉트론이 있다면 다음과 같은 형태일 것이다.

- $ {x_1, x_2, ..., x_i} $ : input

- $ {y_1, y_2, ..., y_k} $ : output

- 첫 hidden layer의 j번째 unit의 output $ z_j $ 은 다음과 같다.(activation function까지 적용한다.) ( $ w_{j0}$ 은 bias )

$$ z_j = h \left( \sum_{i}{w_{ji}x_i + w_{j0}} \right) $$

- output unit의 activation은 다음과 같다.

$$ y_k \ \sum_{j}{w_{kj}z_j + w_{k0}} $$

1. input에 linear transform을 적용

그런데 input data에 linear transform을 적용했다면 input은 다음과 같은 형태가 된다.

$$ x_i = \tilde{x}_i = ax_i + b $$

그렇다면 corresponding linear transformation을 weight, bias에 적용해야 기존과 같은 학습이 된다. 바뀐 weight와 bias 표현식은 다음과 같다.

$$ w_{ji} \rightarrow \tilde{w}_{ji} = {1 \over a} w_{ji} \quad and \quad w_{j0} = w_{j0} - {b \over a} \sum_{i}{w_{ji}} $$

2. output에 linear transform을 적용

만약 output data에 linear transform을 적용했다면 output은 다음과 같은 형태가 된다.

$$ y_k = \tilde{y}_{k} = cy_k + d $$

이런 linear transformation이 적용된 output을 출력하기 위해, 이전 layer인 second layer의 weight와 bias는 다음과 같다.

$$ w_{kj} \rightarrow \tilde{w}_{kj} cw_{kj} \quad and \quad w_{k0} = cw_{k0} + d $$

이렇게 바뀐 weight에는 기존 weight decay를 적용할 수 없다. 기존 식을 회상해 보자.

$$ E(w) = E_0(w) + {1 \over 2}\lambda\sum_{i}{w_{i}^2} $$

이 식은 바뀐 weight의 성분들을 반영하지 못한다. 또한 여러 단계에서 scaling을 적용했다면, 바뀐 weight들을 서로 다르게 취급해야 한다.

만약 위 예시에서 input, output 모두 linear transform을 적용했다면 regularization을 다음과 같이 수정하면 사용할 수 있다.

$$ {{\lambda}_1 \over {2}} \sum_{w \in W_1}{w}^2 + {{\lambda}_2 \over {2}} \sum_{w \in W_2}{w}^2 $$

- $ w_1 $ : first layer의 weights

- $ w_2 $ : second layer의 weights

이 경우 rescaled된 weight로 반영할 수 있다.

$$ \lambda_1 \rightarrow a^{1/2}\lambda_1 \quad and \quad \lambda_2 \rightarrow c^{-1/2}\lambda_2 $$

2. **새로운 오분류를 망각**

또한 대체로 weight decay는 <U>단층 퍼셉트론에서만 쓰인다.</U> 그 이유는 새롭게 오분류된 훈련점들이 가중치 벡터에 너무 큰 영향을 미치는 경우, weight decay에 의해 망각이 너무 빨리 일어나는 경향이 있기 때문이다. 게다가 단층 퍼셉트론은 이후 정리할 다른 종류의 regularization 기법이 더 흔히 쓰인다.

---

## 1.4.1.2 신경망 구조와 매개변수 공유

**RNN**(Recurrent Neural Network, 순환 신경망), **CNN**이 대표적이다. 한 문장을 구성하는 일련의 단어들이 서로 연관되어 있을 때가 많고, 이미지의 인접 픽셀들도 마찬가지로 연관되는 경우가 일반적이다. 따라서 이런 통찰을 이용하면 더 적은 수의 매개변수로 신경망 구조를 만들 수 있다.

---

## 1.4.1.3 Ealry Stopping(조기 종료)

특정 조건을 만족하면 gradient descent의 반복을 일찍 끝내는 기법이다. 종료 시점을 결정하는 일반적인 방법은 다음과 같다.

- 우선 train data set의 일부를 따로 빼서 validation data set을 만든다. 
 
- validation data set을 시험 자료로 써서 model의 error를 측정한다.
 
- validation data의 error가 가장 작은 지점에서 training을 끝낸다.

![early stopping](images/early_stopping.png)

> weight vector는 원점에서 시작해서 $w_{ML}$ 로 나아간다.

Ealry Stopping은 regularization의 대안인데, <U>지속되는 training으로 계속해서 매개변수가 커지는 것을 방지한다</U>는 점에서 하나의 regularization 항으로 작용한다고 할 수 있다.

---

## 1.4.1.4 너비와 깊이의 절충

hidden layer 안에 수많은 hidden unit이 있다면 layer가 두 개인 다층 신경망도 보편적 함수 근사기가 될 수 있다. 하지만 대체로 depth를 늘리면서 hidden unit들을 줄이는 방향으로 설계하는데, 이런 설계는 일종의 regularization에 해당한다. depth를 계속 깊게 만들면서 unit을 줄이다 보면, layer를 늘려서 생긴 매개변수 증가보다도 <U>layer 너비가 줄어들면서 매개변수가 감소한 영향이 더 커지게 된다</U>. 

이 경우 overfitting보다는 낫지만, 또 다른 방향의 문제점을 해결해야 될 수 있다. 신경망의 여러 layer에서 loss function의 미분값 변동이 크면, 적절한 갱신이 이뤄지지 않을 수 있다. 그런 경향은 **gradient vanishing**(기울기 소실)과 **gradient explosion**(기울기 폭발) 현상을 일으킬 수 있다.

> 간단히 vanishing gradient problem(기울기 소실 문제)은 backpropagation이 진행되면서 input 근처 layer의 갱신 크기가 output에 가까운 layer들에 비해 훨씬 작아지는 문제다. 

> 이를 해결하기 위해 기울기가 더 큰 activation function을 사용하고 weight의 크기 자체도 크게 설정할 수 있지만, 정도가 심하면 반대로 기울기가 폭발하는 문제가 발생할 수 있다.

---

## 1.4.1.5 ensemble method(앙상블 방법)

model의 generalization 능력을 키우기 위해 **bagging**(배깅)과 같은 다양한 앙상블 방법이 쓰인다.(이런 방법은 신경망뿐만 아니라 모든 종류의 기계 학습 알고리즘에 적용된다.) 

> Bagging은 Bootstrap Aggregation의 약자다. sample을 여러 번 뽑아(Bootstrap) 각 모델을 학습시켜 결과물을 집계(Aggregation)하는 방법이다.

신경망에 특화된 앙상블 방법도 여럿 있는데, 대표적으로 **dropout**(드롭아웃), **dropconnect**(드롭커넥트)가 있다. 대체로 accuracy를 약 2% 증가시키는 것이 가능하다. 그러나 구체적인 개선 정도는 어떤 자료인가, 어떤 training 방법인가에 달렸다. 예를 들어 hidden layer의 activation을 normalization(정규화)해 버리면 dropout의 효과가 줄어들 수 있다.

---
