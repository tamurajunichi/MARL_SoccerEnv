# MARL_SoccerEnv
マルチエージェント強化学習用のサッカー環境

# Introduction
HalfFieldOffense環境は現状シングルエージェントでの環境を使用したものが多く、マルチエージェントでの環境があまり用意されていません。Ray,Rllibとgymを使用した環境はありますがどちらも複雑になっているため、カスタムして使うには敷居が高い。こんな感じに使えばマルチエージェント化して使えるという一例。
参考元サイトからコードを引用して使用

The HalfFieldOffense environment is currently mostly used as a single-agent environment, so there are not many multi-agent environments available. The multi-agent environment using Ray, Rllib and gyms is too complex to be used as a reference. This is an example of how it can be used as a multi-agent system. The code is quoted and used from the reference site.



# 参考
## 環境
The RoboCup Soccer Simulator:https://github.com/rcsoccersim

Half Field Offense:https://github.com/LARG/HFO

## 強化学習
DDPG:https://github.com/openai/spinningup

PA-DDPG:https://github.com/mhauskn/dqn-hfo

Multi-Pass Deep Q-Networks:https://github.com/cycraig/MP-DQN

Random Network Distillation:https://github.com/openai/random-network-distillation

Curious Explorer:https://github.com/SaeedTafazzol/curious_explorer
