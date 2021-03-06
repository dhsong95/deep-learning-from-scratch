3장: 신경망
=========

1. 퍼셉트론은 신경망
   1. 퍼셉트론과 신경망 노드(뉴런)가 있다. 노드에서는 입력에 대한 가중치와 일련의 연산을 통해 출력이 결정된다.
   2. 신경망에서 입력에 대한 가중합에 적용되는 연산은 활성함수(Activation Fuction)이라고 부른다.
   3. 퍼셉트론은 계단 함수를 활성함수로 사용하는 신경망이라고할 수 있다.
2. 다양한 활성함수(Activation Fuction) [\[코드\]](./activations.py)
   1. 계단함수: 0 보다 크면 1. 아닌 경우 0
   2. 시그모이드 함수(Sigmoid): 어떠한 입력이든 0과 1사이로 분포하도록 하는 함수. 1 / 1 + exp(-x)
   3. ReLU 함수: 0 보다 같거나 작으면 모두 0. 아닌 경우 항등 max(0, x)
   4. Softmax 함수: 주로 출력층에서 사용. 출력 결과를 확률로 해석할 수 있도록 만든다. 
      * 코딩으로 구현할 때에는 exp 연산에 대한 오버플로우 주의. 
      * softmax 함수를 연산량이 많은데 비해 효과는 미미하다. 원래의 데이터가 (2, 2, 4) 일 때 softmax는 이를 (0.25, 0.25, 0.5) 로 변환한다. softmax 함수는 일반적으로 출력층에서 가장 높은 확률을 구하는 것이 목적인데, 이러한 목적에 따르면 softmax 하기 전의 데이터를 가지고 최대값을 찾는 것이 가능하다. 따라서 구현 시 softmax 함수는 생략하기도 한다.
3. 활성함수로는 비선형 함수를 사용한다
   1. 선형함수를 사용하는 경우 아무리 쌓아도 비선형의 효과를 얻을 수 없다
   2. 활성화 함수가 f(x) = cx 라고할 때, 이러한 활성함수를 가진 노드를 가지고 3층의 신경망을 만들더라고 c(c(cx)) 이며, 비선형의 효과를 얻을 수 없다
4. 효과적인 연산을 위해서 배치 연산으로 학습 및 예측한다.