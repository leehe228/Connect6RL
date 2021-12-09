# 육목 강화학습 프로젝트 (DQN + MCTS)
건국대학교 컴퓨터공학부 오픈소스SW프로젝트2 육목 강화학습 프로젝트

by [lminjung99](https://github.com/lminjung99), [leehe228](https://github.com/leehe228)

## 주제

**Deep Reinforcement Learning과 Monte-Carlo Tree Search 알고리즘을 이용한 육목 에이전트 구현 및 결과 분석**

- 육목은 오목과 다르게 특별한 제한 없이 양측에게 완벽하게 공평한 게임이다. 오목과 같이 금지된 수가 없을 뿐더러 규칙도 간단하여 컴퓨터과학 연구에 널리 사용되고 있는 게임이다.
- 하지만, 바둑과 같이 경우의 수가 매우 크기 때문에 일반적인 수학 기반 강화학습 알고리즘으로는 학습이 어렵다. 그러므로 딥러닝 기반 강화학습(DRL) 알고리즘을 사용해야 한다.
- 그 예시로, 구글 딥마인드에서 연구한 알파고는 강화학습과 MCTS (Monte-Carlo Tree Search) 알고리즘을 사용해 좋은 성과를 이루어냈다.
- 최근의 여러 연구를 살펴보면, 육목 뿐만 아니라 경우의 수가 매우 많은 문제에서 딥러닝 뿐만 아니라 MCTS와 같은 고전적 수학 방식의 Heuristic Search 기법도 함께 사용되고 있다.
- DQN은 Behavior Policy에서 주로 epsilon-greedy 정책을 채용하는데, 여기서 탐험 정책인 Exploration의 경우 완전 랜덤 탐색 대신 MCTS에 따라 움직이게 된다. 이를 위해, UCT 값 (각 노드의 예상 점수를 계산한 수학적 공식에 의한 특정 값), 노드 방문 횟수, simulation 기능 등을 추가로 개발하여야 한다.
- MCTS를 사용하면 완전 랜덤 정책에서 벗어나 좋은 수의 샘플을 사용해 학습을 진행해 학습 효율을 높일 수 있으며, 시뮬레이션 과정에서 원하는 어느 시간에 중지하더라도 그 시뮬레이션 transaction과 관계 없이 가장 좋은 수를 반환하게 되어 Exploration에서도 적당한 랜덤이 가미된 좋은 수들을 위주로 학습을 진행할 수 있다.
- DQN은 대표적인 off-policy 강화학습 알고리즘으로, Behavior Policy와 Target Policy가 다르기 때문에 병렬 학습 또는 적대적인 학습이 가능하다. 이 장점을 이용해, 학습은 병렬로, 또한 한 게임에서 같은 Target Policy를 가진 에이전트가 경쟁하게 하여 적대적으로 학습시킴으로써 학습 효율을 올릴 것이다.

⇒ 대표적인 딥러닝 기반 강화학습 알고리즘인 Deep Q-Network(DQN)과 서로에게 공평한 게임에서 주로 사용되는 대표적인 휴리스틱 탐색 기법인 MCTS 알고리즘을 결합해 적대적으로 학습시킨 후, 가능한 조합에 대하여 학습 결과와 성능을 비교 및 분석하여 본다.

## 주제 범위

- Python3과 PyGame을 이용해 육목 Environment를 제작하고 Visualization한다.
- Tensorflow 를 이용해 육목 환경에서 학습할 수 있는 DQN 기반 Agent와 Model을 제작한다.
- DQN 알고리즘에 MCTS를 결합한 새로운 알고리즘 모델을 제작한다.
- (1) DQN, (2) DQN + MCTS 두 알고리즘 모델에 대해 학습을 진행하고, 여러 관점에서 성능을 측정하고 결과를 분석해본다.
- 추가적으로, 사람과 대결할 수 있는 Application을 제작한다. (trained model을 활용)

## Environment

**MDP of the Connect6 Environment**

$$[<S, A, P, R, \gamma>]$$

$$[S]$$ : $19 \times 19$ 테이블 위 각 칸은 비어있거나 검은 돌이 놓여있거나 흰 돌이 놓여있다. 그러므로 state의 개수는 $C^3_1 \times 19\times 19=1083$개이다.

$A$ : 총 $361$개의 칸 위에 돌을 놓는 것이 action이므로, 총 361개의 descrete한 action

$P$ : $P^a_{ss'}=1 ( \forall a, \forall s, \forall s')$

$R$ : 승리한 state에 대해 +10.0 / 패배한 state에 대해 -10.0 / 중복수에 대해 -0.1 / 모든 스텝에 대해 +0.0001

$\gamma \in [0.05, 0.95]$ : discount factor 

환경의 MDP는 어떤 action이나, 외부의 영향으로 변하지 않으므로 stationary하다.

다음 상태($s'$)는 현재 상태($s$)에서의 행동($a$)에 의해서만 결정되므로 markov하다.

⇒ 환경이 stationary하고 markov하므로 강화학습으로 해결할 수 있는 문제이다.

## Requirements

**Major Functional Requirements**

1. Visualized Environment
    
    PyGame 라이브러리를 이용해 시각화한다. 보드 위에 각 수에 따른 돌이 보여지도록 한다. 환경의 경우 규칙에 따른 reward($R_s^a$)를 규정하고, 전이 함수를 정의한다. ($s$ → $a$ → $s’$)
    
2. Heuristic Layer & Heuristic Agent
    
    각 칸에 대해 해당 칸에 돌을 놓는다면 생기는 일렬의 돌의 수 중 최댓값을 휴리스틱 값으로 사용한다.
    
    성능 비교용 Heuristic Agent는 이 heuristic 값에 따라 돌을 놓는다.
    
    input으로 들어가는 2개의 layer는 나의 입장, 상대편의 입장 두 개의 layer가 추가된다 (돌을 놓으면 생기는 돌의 수 중 최댓값, 상대방의 경우 돌을 놓으면 막아지는 돌 중 최댓값)
    
3. DQN Algorithm Model
    
    기존 DQN 논문을 참조하여 육목 알고리즘에 알맞는 DQN 모델을 개발한다. PyTorch를 이용하며, Network의 Hidden Layer 수, Layer Dimension을 포함한 hyperparameter들은 등은 추후 테스트 과정에서 (필요하다면 hyperparameter optimization을 통해) 조절한다. Behavior Policy는 epsilon-greedy, Target Policy는 greedy 정책을 선택한다. 또한, TensorboardX를 통해 학습 추이를 추적할 수 있도록 logger 기능을 포함해 개발한다.
    
4. DQN + MCTS Model
    
    2번 과정에서 제작한 DQN 모델을 바탕으로 Behavior Policy의 Exploration 부분을 epsilon-greedy가 아닌 Monte-Carlo Tree Search 탐색 기법으로 변경한다.
    
5. 사람과 대결할 수 있는 육목 Application
    
    1번 과정에서 제작한 Visualized Environment에서 사용자가 수를 선택할 수 있는 기능을 추가한다.
    

## Architecture

**현재 구현된 Architecture (2 agents, adversarial)**

![](https://user-images.githubusercontent.com/37548919/145352652-94dcc2b4-c434-4015-89e3-25ba51948158.png)

**변경 예정인 Architecture (1 agent)**

학습 능률을 증가시키고, reward를 통해 학습 추이를 추정할 수 있도록 하기 위함

![](https://user-images.githubusercontent.com/37548919/145352786-e193bd79-d084-4222-b92c-3d64b959794a.png)

**Architecture of Q Network**

![](https://user-images.githubusercontent.com/37548919/145352800-55e33d10-722a-4740-b5e0-2dcd3c9910e0.png)

→ CNN Layers

→ Fully Connected Layers

→ Input은 현재 board의 state만 들어가나, heuristic layer를 추가할 예정

### 최종 예상산출물

- DQN 알고리즘 모델과 trained data (loss, reward 등)
- DQN + MCTS 모델과 trained data (loss, reward 등)
- 학습된 모델을 이용한 성능 측정 결과 (승패, reward, 승리까지 소요 시간 등)
- 사람과 플레이할 수 있는 executable program
