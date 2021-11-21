#!/bin/bash

for x in {1..20}
do
  echo $x"回目の実行"
  /home/notolab/anaconda3/envs/hfo/bin/python /home/notolab/git/MARL_SoccerEnv/main.py
done