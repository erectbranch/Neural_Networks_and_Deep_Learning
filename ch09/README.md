# 9 심층 강화 학습

복잡한 환경과 상호작용하면서 **reward**(보상)을 얻음으로써, 뭔가를 배우는 reward 주도적 시행 착오 과정을 ML 분야에서 **reinforcement learning**(강화 학습)이라고 지칭한다.

> 최근 몇 년 동안 reinforcement learning model을 적용한 비디오 게임 플레이 model이나, 정상급 바둑 기사를 이기는 AlphaGo 등의 주목할 결과들이 많이 나왔다. 

>이외에도 자동차에 장착된 여러 감지기의 신호를 바탕으로 스스로 운전하는 자율주행(보통은 supervised learning을 쓰지만 적용할 수 있는 가능성이 크다.)이나, self-learning(자기 학습) 로봇 등 다양한 분야에서 사용되고 있다.

reinforcement learning은 <U>평가는 간단하지만, 명시는 어려운 과제</U>에 적합하다. 예를 들어 체스와 같은 게임이 끝나면 성과를 평가하는 것은 쉽지만, 매 상황마다 구체적인 최적의 수를 명시하기는 어렵다.

reinforcement learning는 reward를 정의해 주기만 하면, 그러한 reward를 최대화하는 행동을 agent가 스스로 터득한다. 즉, reinforcement learning는 <U>복잡한 행동의 training을 단순화</U>하는 수단으로 사용한다. 

> 마치 **ant hypothesis**(개미 가설)과 비슷하다. "행동 시스템 관점에서 인간은 굉장히 단순하다. 인간 행동의 외견에서 보이는 복잡성은, 시간의 경과에 따라 인간이 처한 환경의 복잡성이 반영된 것뿐이다."

복잡한 과제를 더 작은 partition으로 분해하는 대신 더 단순한 reward 관점에서 바라본다는 점에서, reinforcement learning은 본질적으로 **end-to-end system**(종단간 시스템)에 해당된다.

---

## 9.1 state 없는 algorithm: multi-armed bandit problem

reinforcement learning의 성격을 보여주는 가장 간단한 예는 **multi-armed bandit problem**(여러 팔 강도 문제)이다. 이 문제에서 도박꾼은 여러 슬롯 머신의 기대 보상이 서로 다르지만, 도박꾼은 이 기대 보상을 미리 알지 못한다. 따라서 여러 슬롯 머신을 실제로 돌려 보는 **exploration**(탐색) 과정을 거쳐야 한다. 또한 exploration으로 얻은 정보를 바탕으로 이를 **exploitation**(활용)해야 한다.

> multi-armed bandit(여러 팔 강도)이라는 이름은 길쭉한 레버(arm)을 당기면 슬롯이 돌아가는 형태의 슬롯 머신에 붙은 별명이다.

하지만 슬롯 머신을 무작위로 exploration하는 과정은 돈을 잃을 가능성이 있다. 따라서 multi-armed bandit problem는 exploration과 exploitation의 균형을 찾는 문제라고 할 수 있다. 

슬롯 머신은 state가 없고 매 결정이 independent한 단순한 예시지만, 실제 응용에서는 훨씬 복잡한 경우를 다루게 된다.

> 비디오 게임이나 자율 주행차의 여러 감지기의 설정은 훨씬 복잡하다. 화면 pixel들이나 도로 상황들로 정의되는 system **state**를 고려해야 한다.

그렇다면 exploration과 exploitation의 최선의 절충점을 어떻게 찾아내야 할까? 이 문제에서는 <U>state가 존재하지 않으므로</U> 한 시행이 제공하는 reward가, 그 이전 시행들과 동일한 probability distribution(확률 분포)를 따른다고 가정한다.

exploration과 exploitation을 절충할 수 있는 strategy은 여러 가지가 있다.

---

### 9.1.1 simple algorithm

우선 각 슬롯 머신을 고정된 횟수로 exploration한다. 그 다음 exploitation 단계로 넘어가서 가장 나은 reward의 슬롯 머신 하나만을 계속 시행할 수 있다. 언뜻 생각하면 합리적인 strategy 같지만 몇 가지 단점을 가지고 있다.

1. exploration 단계에서 각 슬롯 머신을 시행할 횟수를 결정하기는 쉽지 않다. 또한 슬롯 머신의 reward 수준을 평가하기 위해 긴 시간이 필요할 수도 있다.

  > 특히 지급 사건(특정 슬롯 조합이 나와서 돈을 따는 경우)이 비지급 사건보다 드물게 발생하면 더 시간이 필요하다.

2. 만약 strategy가 잘못 되어 reward가 낮은 슬롯 머신이 선택되었다면, exploitation 단계에서 계속 reward가 낮은 슬롯 머신만 계속 돌리게 될 것이다.

---

### 9.1.2 ε-greedy algorithm

ε-greedy algorithm은 시행을 크게 낭비하지 않으면서 최적의 strategy를 사용하기 위한 방법이다. 전체 시행 중 비율 ε만큼만의 시행을 exploration에 사용한다. 

- exploitation 시행도 전체 시행 중에서 무작위로(ε 확률로) 선택한다.

- 전체의 (1-ε) 비율에 해당하는 나머지 exploitation 시행들은, 그때까지 평균 지급액이 가장 큰 슬롯 머신을 돌리는 데 사용한다.

따라서 exploration 시행들과 exploitation이 섞인 형태가 된다. exploitation 단계가 일찍 시작되기 때문에, 전체 과정 중 상당 부분에서 최적의 strategy 전략을 사용할 가능성이 크다.

> 또한 잘못된 strategy에 영원히 발목을 잡힐 일이 없다.

parameter ε는 보통 0.1로 두지만, 최적의 ε 값은 주어진 응용마다 다르다. 실제로 이 ε 값을 구하는 것이 어려울 때도 많은데, 대체로 ε 값을 작게 잡을 필요가 있다는 점만큼은 확실하다. 단, ε 값이 작으면 좋은 슬롯 머신을 찾아내기까지의 시간이 오래 걸리게 된다.

따라서 ε를 큰 값에서 시작해서 점차 줄여나가는 **annealing**(정련) 기법이 흔히 쓰인다.

---

### 9.1.3 upper bounding method

dynamic한 설정에서 ε-greedy algorithm이 효율적이기는 하지만, 새 슬롯 머신의 reward 수준을 train하는 데는 여전히 비효율적이다.

**upper bounding method**(상계 방법)은  아직 충분히 시험해 보지 않은 슬롯 머신을 더 낙관적으로 취급하는 방식을 사용한다. ε-greedy algorithm이 기준을 평균 지급액을 사용한 것과 달리, upper bounding method는 지급액의 **statistical upper bound**(통계적 상계)가 가장 큰 슬롯 머신을 선택한다.

$$ U_{i} = Q_{i} + C_{i} $$

- $U_{i}$ : $i$ 번째 슬롯 머신의 지급액 upper bound

- $Q_{i}$ : 총 기대 보상

- $C_{i}$ : 단측(한쪽) 신뢰구간의 길이

여기서 $C_{i}$ 값은 슬롯 머신의 불확정성을 보충하는 일종의 보너스로 작용한다. $C_{i}$ 값은 지금까지의 시행의 mean reward의 standard deviation(표준 편차)와 비례한다.

> central limit theorem(중심극한정리)에 따라, 이 standard deviation ${\sigma}_{i}$ 은 슬롯 머신 $i$ 의 시행 횟수의 제곱근 $\sqrt{n_{i}}$ 과 반비례하게 된다.

$$ C_{i} = K \cdot {\sigma}_{i} / \sqrt{n_{i}} $$

- $K$ : 신뢰구간의 길이에 영향을 주는 계수

- ${\sigma}_{i}$ : $i$ 번째 슬롯 머신의 standard deviation

- ${n}_{i}$ : $i$ 번째 슬롯 머신의 시행 횟수 

아직 많이 시행되지 않은 슬롯 머신은 신뢰구간이 길어져서 그 upper bound가 커지며, 결과적으로 좀 더 자주 시행되게 된다.

이처럼 upper bounding method는 exploration과 exploitation이 구분되지 않고 두 측면을 모두 반영하는 이중의 효과를 낸다. exploration과 exploitation의 절충은 해당 시행 개수가 아니라, 확률적 신뢰도 구간을 통해 제어한다.

예를 들어 $K=3$ 으로 두면 normal distribution의 upper bound에 99.99% 신뢰 구간을 사용하는 것이 된다. 일반적으로 $K$ 를 크게 할수록 불확실성을 표현하는 $C_{i}$ 가 커지며, 더 exploration에 치중하게 된다.

---
