#!/bin/bash

export HF_HOME="/vol/bitbucket/cp524/hf_cache"

python launch_eval_runs_meissonic.py \
  --use_smc \
  --model_name='meissonic-fp16-monetico' \
  --lmbda=2.0 \
  --resample_frequency=10 \
  --resample_t_start=10 \
  --resample_t_end=80 \
  --num_particles=16 \
  --potential_type=max \
  --guidance_reward_fn='ImageReward'
