#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=resgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cp524

export HF_HOME="/vol/bitbucket/cp524/hf_cache"

# for offline loading only
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Activate virtual environment
export PATH=/vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/venv/bin:$PATH
source /vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/venv/bin/activate

# Set up CUDA
source /vol/cuda/12.5.0/setup.sh

# Navigate to script directory
cd /vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/text_to_image

export PYTHONUNBUFFERED=1

# Run training notebook
MODEL="meissonic-fp16-monetico"
COMMON_ARGS=(
  --model_name="$MODEL"
  --num_inference_steps=100
  --resample_frequency=10
  --resample_t_start=10
  --resample_t_end=90
  --potential_type=max
  --guidance_reward_fn=ImageReward
  --metrics_to_compute='ImageReward#HumanPreference'
  --use_fkd_log_impl
  --use_smc
  --max_decode_batch_size=16
)


for phi in 4 1; do
  for LAMBDA in 50.0 10.0 4.0 2.0; do
    for n in 2 4 8 16; do
      if [[ "$phi" == 4 && "$LAMBDA" == 50.0 && "$n" == 2 ]]; then
        continue
      fi
      echo "▶ Running SMC with ${n} particles, φ=${phi}, λ=${LAMBDA}"
      python launch_eval_runs_meissonic.py \
        "${COMMON_ARGS[@]}" \
        --num_particles=$n \
        --lmbda=$LAMBDA \
        --num_x0_samples=$phi
    done
  done
done
