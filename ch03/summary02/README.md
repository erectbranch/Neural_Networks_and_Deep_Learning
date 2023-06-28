## 3.6 Batch Normalization(배치 정규화)

**Batch Normalization**(배치 정규화)는 gradient vanishing 및 explosion 문제를 해결하기 위해 제안된 방법이다.

- **internal covariate shift**(내부 공변량 이동)에 초점을 맞춘다.

> covariate(공변량)이란 독립변수 외 종속변수에 영향을 줄 수 있는 잡음인자 변량을 의미한다.

> 보통 연구에서는 여러 독립변수가 종속변수에 얼마나 영향을 주는지 확인하려고 하는데, 잡음인자가 있을 경우 독립변수의 순수한 영향력을 알 수 없다.

---

### 3.6.1 covariate shift

아래 그림은 **covariate shift**(공변량 이동)을 나타낸다.

![covariate shift](images/covariate_shift.png)

- train/test dataset distribution이 다르다.

흔히 훈련한 모델을 test dataset으로 성능을 측정했을 때 잘 안나오는 경우 overfitting을 의심하지만, covariate shift 관점에서 보면 dataset을 자체를 잘못 구성했기 때문에 발생했을 수 있다.

가령 게임 렌더링으로 구성된 도로에서 training을 수행한 자율주행 자동차가 있다고 하자. train dataset이 너무 단순한 텍스처로만 이루어져 있기 때문에 실제 복잡한 도로에서는 잘 동작하지 않을 것이다. 또한 낮/밤의 차이만으로도 정확도 차이가 크게 날 수 있다.

---

### 3.6.2 internal covariate shift

> [internal covariate shift 정리](https://wegonnamakeit.tistory.com/47)

아래 그림은 **internal covariate shift**(내부 공변량 이동)을 나타낸다.

![internal covariate shift](images/internal_covariate_shift.png)

- activation distribution이 달라진다.

---

### 3.6.3 Normalization Layer

BN(Batch Normalization)은 기본적으로 hidden layer 사이에 **normalization layer**(정규화층)을 추가해서 구현한다. activation은 다음과 같은 mean, std를 갖는 정규분포 형태가 된다.

- ${\beta}_{i}$ : mean(평균)

- ${\gamma}_{i}$ : std(표준편차)

위 두 parameter는 training 과정에서 학습된다. 주의할 점은 <U>mini-batch</U> 단위마다 고유한 ${\beta}_{i}$ , ${\gamma}_{i}$ 를 사용한다는 점이다.

> 하지만 이러한 변형은 지난 layer에서 얻은 표현력을 감소시킬 수 있다.

> 특히 모든 data를 단순히 평균 0, 표준편차 1로 변형시키는 **whitening** 방식을 적용하면 이런 문제가 발생하게 된다.

---

