<div width="100%" height="100%" align="center">
  
<h1 align="center">
  <p align="center">신경망과 심층학습</p>
  <a href="https://jpub.tistory.com/960">
    <img width="50%" src="cover.jpg" />
  </a>
</h1>
  
  
<b>차루 C. 아가르왈 저 · 류광 번역</b></br>
제이펍 · 2019년 09월 17일 출간</br>
[[강의 자료](https://www.dropbox.com/sh/k3wa8wll99oznhp/AAA2QWTsxoCISYu_bPDP_S7ta?dl=0)] | [[해답집](https://t1.daumcdn.net/cfile/tistory/99B840415D80309C2D)] | [[정오표](https://jpub.tistory.com/961)]</b>

</div>

## :bulb: 목표

- **SIMPLE: 뉴럴 네트워크와 딥러닝 개념을 정리한다.**

  > 뉴럴 네트워크와 딥러닝 개념을 공부&정리하는 것에 초점을 둔다.

</br>

## 🚩 정리한 문서 목록

### 📔 신경망 입문

- **통상적인 완전 연결 순방향 신경망**

  > [통상적인 완전 연결 순방향 신경망 개념](https://github.com/erectbranch/Neural_Networks_and_Deep_Learning/tree/master/ch01/summary01): 퍼셉트론 개요, 가중치 갱신, 목적/손실/활성화 함수의 종류와 선택, 활성화 함수의 미분, 다층 신경망 개요, 계산 그래프 개념, Backpropagation

  > [신경망 훈련의 문제점, 신경망 합수 합성의 강점](https://github.com/erectbranch/Neural_Networks_and_Deep_Learning/tree/master/ch01/summary02): Overfitting과 해결책 - Regularization / 매개변수 공유 / Early stopping / 너비와 깊이의 절충 / Ensemble, 기울기 소실/폭발, 수렴의 어려움, local optimum, 신경망 함수 합성의 강점

- **여러 신경망 구조**

<br/>

## :mag: 목차

### 1장 신경망 입문

1.1 소개 1

- 1.1.1 인간 대 컴퓨터: 인공지능의 한계 확장 4

1.2 신경망의 기본 구조 6

- 1.2.1 단일 계산층: 퍼셉트론 7

- 1.2.2 다층 신경망 25

- 1.2.3 계산 그래프로서의 다층망 28

1.3 역전파를 이용한 신경망 훈련 30

1.4 신경망 훈련의 실질적인 문제점들 35

- 1.4.1 과대적합 문제점 35

- 1.4.2 기울기 소실 및 폭발 문제 41

- 1.4.3 수렴의 어려움 41

- 1.4.4 국소 가짜 최적해 42

- 1.4.5 계산의 어려움 43

1.5 함수 합성이 강력한 이유 44

- 1.5.1 비선형 활성화 함수의 중요성 47

- 1.5.2 깊이를 이용한 매개변수 요구수준 감소 49

- 1.5.3 통상적이지 않은 신경망 구조들 51

1.6 흔히 쓰이는 신경망 구조들 54

- 1.6.1 얕은 모형으로 기본적인 기계 학습 흉내 내기 54

- 1.6.2 방사상 기저 함수(RBF) 신경망 54

- 1.6.3 제한 볼츠만 기계 55

- 1.6.4 순환 신경망 56

- 1.6.5 합성곱 신경망 59

- 1.6.6 위계적 특징 공학과 미리 훈련된 모형 61

1.7 고급 주제 64

- 1.7.1 강화 학습 64

- 1.7.2 자료 저장과 계산의 분리 65

- 1.7.3 생성 대립 신경망(GAN) 66

1.8 주요 벤치마크 두 가지 67

- 1.8.1 필기 숫자들을 담은 MNIST 데이터베이스 67

- 1.8.2 ImageNet 데이터베이스 69


### 2장 얕은 신경망을 이용한 기계 학습

2.1 소개 79

2.2 이진 분류 모형을 위한 신경망 구조 82

- 2.2.1 퍼셉트론 다시 보기 83

- 2.2.2 최소제곱 회귀 85

- 2.2.3 로지스틱 회귀 91

- 2.2.4 지지 벡터 기계 94

2.3 다중 분류 모형을 위한 신경망 구조들 97

- 2.3.1 다부류 퍼셉트론 97

- 2.3.2 웨스턴-왓킨스 SVM 99

- 2.3.3 다항 로지스틱 회귀(소프트맥스 분류기) 101

- 2.3.4 다중 분류를 위한 위계적 소프트맥스 103

2.4 해석성과 특징 선택을 위한 돌출 요인 역전파 104

2.5 자동부호기를 이용한 행렬 인수분해 105

- 2.5.1 자동부호기의 기본 원리 106

- 2.5.2 비선형 활성화 함수 113

- 2.5.3 심층 자동부호기 116

- 2.5.4 이상치 검출에 응용 119

- 2.5.5 은닉층이 입력층보다 넓은 경우 120

- 2.5.6 기타 응용 122

- 2.5.7 추천 시스템: 행 색인과 행 가치 예측 124

- 2.5.8 논의 128

2.6 word2vec: 단순 신경망 구조의 한 응용 129

- 2.6.1 연속 단어 모음을 이용한 신경망 단어 내장 130

- 2.6.2 스킵그램 모형을 이용한 신경망 내장 134

- 2.6.3 word2vec(SGNS)은 로그 행렬 인수분해이다 142

- 2.6.4 보통의 스킵그램은 다항 행렬 인수분해이다 145

2.7 그래프 내장을 위한 간단한 신경망 구조 146

- 2.7.1 임의의 간선 횟수 처리 148

- 2.7.2 다항 모형 149

- 2.7.3 DeepWalk 및 node2vec의 관계 149


### 3장 심층 신경망의 훈련

3.1 소개 157

3.2 아주 상세한 역전파 알고리즘 160

- 3.2.1 역전파와 계산 그래프 추상 160

- 3.2.2 해결책은 동적 계획법 166

- 3.2.3 활성화 후 변수를 이용한 역전파 167

- 3.2.4 활성화 전 값을 이용한 역전파 171

- 3.2.5 여러 활성화 함수의 갱신 공식 174

- 3.2.6 벡터 중심적 역전파의 분리 관점 176

- 3.2.7 다중 출력 노드 및 은닉 노드의 손실함수 179

- 3.2.8 미니배치 확률적 경사 하강법 180

- 3.2.9 역전파에서 가중치 공유를 처리하는 요령 183

- 3.2.10 기울기 계산의 정확성 확인 184

3.3 설정과 초기화 문제 186

- 3.3.1 초매개변수 조정 186

- 3.3.2 특징 전처리 188

- 3.3.3 초기화 191

3.4 기울기 소실 및 폭발 문제 193

- 3.4.1 기하학으로 살펴본 기울기 비의 효과 194

- 3.4.2 활성화 함수의 선택을 이용한 부분적인 해법 196

- 3.4.3 뉴런의 죽음과 ‘뇌손상’ 198

3.5 경사 하강 전략들 199

- 3.5.1 학습 속도 감쇄 200

- 3.5.2 운동량 기반 학습 201

- 3.5.3 매개변수 고유 학습 속도 204

- 3.5.4 절벽과 고차 불안정성 210

- 3.5.5 기울기 절단 211

- 3.5.6 2차 미분 212

- 3.5.7 폴리액 평균 224

- 3.5.8 극소점과 가짜 최소점 225

3.6 배치 정규화 226

3.7 가속과 압축을 위한 실질적인 요령들 232

- 3.7.1 GPU 가속 232

- 3.7.2 병렬 및 분산 구현 235

- 3.7.3 모형 압축을 위한 알고리즘 요령 237

### 4장 일반화 능력을 위한 심층 학습 모형의 훈련

4.1 소개 249

4.2 편향 대 분산 절충 관계 256

- 4.2.1 공식적인 관점 258

4.3 모형의 조정 및 평가와 관련된 일반화 문제점 262

- 4.3.1 예비와 교차 검증을 이용한 모형 평가 264

- 4.3.2 자료 집합의 규모에 따른 훈련상의 문제점 266

- 4.3.3 자료 추가 수집 필요성 판정 266

4.4 벌점 기반 정칙화 267

- 4.4.1 잡음 주입과의 관계 269

- 4.4.2 정칙화 271

- 4.4.3 정칙화 대 정칙화 272

- 4.4.4 은닉 단위에 대한 벌점: 희소 표현 학습 273

4.5 앙상블 방법 274

- 4.5.1 배깅과 부표집 275

- 4.5.2 매개변수 기반 모형 선택과 평균화 277

- 4.5.3 무작위 연결 생략 278

- 4.5.4 드롭아웃 278

- 4.5.5 자료 섭동 앙상블 283

4.6 조기 종료 284

- 4.6.1 분산의 관점에서 본 조기 종료 285

4.7 비지도 사전훈련 286

- 4.7.1 비지도 사전학습의 변형들 290

- 4.7.2 지도 사전훈련은 어떨까? 291

4.8 연속법과 커리큘럼 학습 293

- 4.8.1 연속법 294

- 4.8.2 커리큘럼 학습 295

4.9 매개변수 공유 296

4.10 비지도 학습의 정칙화 298

- 4.10.1 값 기반 벌점: 희소 자동부호기 298

- 4.10.2 잡음 주입: 잡음 제거 자동부호기 299

- 4.10.3 기울기 기반 벌점: 축약 자동부호기 301

- 4.10.4 은닉 확률 구조: 변분 자동부호기 305


### 5장 방사상 기저 함수 신경망

5.1 소개 321

5.2 RBF 망의 훈련 326

- 5.2.1 은닉층의 훈련 326

- 5.2.2 출력층의 훈련 328

- 5.2.3 직교 최소제곱 알고리즘 331

- 5.2.4 완전 지도 학습 332

5.3 RBF 망의 변형 및 특수화 333

- 5.3.1 퍼셉트론 판정기준을 이용한 분류 334

- 5.3.2 경첩 손실함수를 이용한 분류 334

- 5.3.3 RBF 망의 선형 분리가능성 개선 335

- 5.3.4 RBF 망을 이용한 보간 337

5.4 핵 방법들과의 관계 338

- 5.4.1 특수한 RBF 망으로서의 핵 회귀 338

- 5.4.2 특수한 RBF 망으로서의 핵 SVM 339

- 5.4.3 관찰 340


### 6장 제한 볼츠만 기계

6.1 소개 345

- 6.1.1 역사적 관점 346

6.2 홉필드 망 347

- 6.2.1 훈련된 홉필드 망의 최적 상태 구성 349

- 6.2.2 홉필드 망의 훈련 352

- 6.2.3 간단한 영화 추천 시스템의 구축과 그 한계 354

- 6.2.4 홉필드 망의 표현력 증가 355

6.3 볼츠만 기계 357

- 6.3.1 볼츠만 기계의 자료 생성 359

- 6.3.2 볼츠만 기계의 가중치 학습 360

6.4 제한 볼츠만 기계 363

- 6.4.1 RBM의 훈련 366

- 6.4.2 대조 발산 알고리즘 367

- 6.4.3 실천상의 문제와 알고리즘 수정 369

6.5 제한 볼츠만 기계의 응용 370

- 6.5.1 RBM을 이용한 차원 축소와 자료 재구축 371

- 6.5.2 RBM을 이용한 협업 필터링 374

- 6.5.3 RBM을 이용한 분류 378

- 6.5.4 RBM을 이용한 주제 모형화 383

- 6.5.5 RBM을 이용한 다중 모드 자료 기계 학습 385

6.6 RBM을 이진 자료 이외의 자료에 적용 387

6.7 중첩된 RBM 389

- 6.7.1 비지도 학습 392

- 6.7.2 지도 학습 392

- 6.7.3 심층 볼츠만 기계와 심층 믿음망 393


### 7장 순환 신경망

7.1 소개 399

- 7.1.1 순환 신경망의 표현력 403

7.2 순환 신경망의 구조 404

- 7.2.1 RNN을 이용한 언어 모형 예제 408

- 7.2.2 시간에 따른 역전파 411

- 7.2.3 양방향 순환 신경망 415

- 7.2.4 다층 순환 신경망 418

7.3 순환 신경망 훈련의 어려움과 그 해법 420

- 7.3.1 층 정규화 424

7.4 반향 상태 신경망 426

7.5 장단기 기억(LSTM) 429

7.6 게이트 제어 순환 단위(GRU) 433

7.7 순환 신경망의 응용 436

- 7.7.1 자동 이미지 캡션 생성 437

- 7.7.2 순차열 대 순차열 학습과 기계 번역 439

- 7.7.3 문장 수준 분류 444

- 7.7.4 언어적 특징을 활용한 토큰 수준 분류 445

- 7.7.5 시계열 예상 및 예측 447

- 7.7.6 시간적 추천 시스템 450

- 7.7.7 2차 단백질 구조 예측 453

- 7.7.8 종단간 음성 인식 453

- 7.7.9 필기 인식 454


### 8장 합성곱 신경망

8.1 소개 461

- 8.1.1 역사 및 생물학의 영향 462

- 8.1.2 좀 더 넓은 관점에서 본 합성곱 신경망 464

8.2 합성곱 신경망의 기본 구조 465

- 8.2.1 여백 채우기 472

- 8.2.2 보폭 474

- 8.2.3 전형적인 설정 475

- 8.2.4 ReLU 층 476

- 8.2.5 풀링 477

- 8.2.6 층들의 완전 연결 479

- 8.2.7 서로 다른 층들의 교대 구성 480

- 8.2.8 국소 반응 정규화 484

- 8.2.9 위계적 특징 공학 485

8.3 합성곱 신경망의 훈련 487

- 8.3.1 합성곱 층에 대한 역전파 487

- 8.3.2 역/전치 필터를 이용한 합성곱 연산으로서의 역전파 489

- 8.3.3 행렬 곱셈으로서의 합성곱 및 역전파 490

- 8.3.4 자료 증강 493

8.4 합성곱 신경망 구조의 사례 연구 495

- 8.4.1 AlexNet 495

- 8.4.2 ZFNet 499

- 8.4.3 VGG 500

- 8.4.4 GoogLeNet 504

- 8.4.5 ResNet 507

- 8.4.6 깊이의 효과 512

- 8.4.7 미리 훈련된 모형들 512

8.5 시각화와 비지도 학습 514

- 8.5.1 훈련된 합성곱 신경망의 특징 시각화 515

- 8.5.2 합성곱 자동부호기 522

8.6 합성곱 신경망의 응용 529

- 8.6.1 내용 기반 이미지 검색 530

- 8.6.2 물체 위치 추정 530

- 8.6.3 물체 검출 532

- 8.6.4 자연어 처리와 순차열 학습 534

- 8.6.5 동영상 분류 535


### 9장 심층 강화 학습

9.1 소개 543

9.2 상태 없는 알고리즘: 여러 팔 강도 547

- 9.2.1 단순한 알고리즘 548

- 9.2.2 탐욕 알고리즘 548

- 9.2.3 상계 방법 549

9.3 강화 학습의 기본 틀 550

- 9.3.1 강화 학습의 어려움 553

- 9.3.2 틱택토 게임을 위한 간단한 강화 학습 554

- 9.3.3 심층 학습의 역할과 잠정적 알고리즘 555

9.4 가치 함수 학습의 부트스트래핑 558

- 9.4.1 함수 근사기로서의 심층 강화 학습 모형 560

- 9.4.2 응용 사례: 아타리 설정을 위한 강화 학습 심층 신경망 565

- 9.4.3 정책 내 방법 대 정책 외 방법: SARSA 566

- 9.4.4 상태의 모형화 대 상태-동작 쌍 568

9.5 정책 기울기 방법 571

- 9.5.1 유한차분법 573

- 9.5.2 가능도비 방법 574

- 9.5.3 지도 학습과 정책 기울기 방법의 결합 577

- 9.5.4 행위자-비평자 방법 578

- 9.5.5 연속 동작 공간 580

- 9.5.6 정책 기울기 방법의 장단점 581

9.6 몬테카를로 트리 검색 582

9.7 사례 연구 584

- 9.7.1 알파고: 세계 최고 수준의 인공지능 바둑 기사 584

- 9.7.2 스스로 배우는 로봇 591

- 9.7.3 대화 시스템 구축: 챗봇을 위한 심층 학습 596

- 9.7.4 자율주행차 600

- 9.7.5 강화 학습을 이용한 신경망 구조의 추론 603

9.8 안전과 관련된 실무적인 어려움들 604


### 10장 심층 학습의 고급 주제들

10.1 소개 613

10.2 주의 메커니즘 616

- 10.2.1 시각적 주의의 순환 모형 618

- 10.2.2 기계 번역을 위한 주의 메커니즘 622

10.3 외부 메모리가 있는 신경망 627

- 10.3.1 가상 정렬 게임 628

- 10.3.2 신경 튜링 기계 631

- 10.3.3 미분 가능 신경 컴퓨터 개괄 639

10.4 생성 대립 신경망(GAN) 641

- 10.4.1 생성 대립 신경망의 훈련 642

- 10.4.2 변분 자동부호기와 비교 646

- 10.4.3 GAN을 이용한 이미지 자료 생성 647

- 10.4.4 조건부 생성 대립 신경망 649

10.5 경쟁 학습 655

- 10.5.1 벡터 양자화 657

- 10.5.2 코호넨 자기조직화 지도 658

10.6 신경망의 한계 662

- 10.6.1 대담한 목표: 단발 학습 662

- 10.6.2 대담한 목표: 에너지 효율적 학습 665

