#!/bin/bash

export HF_HOME="/vol/bitbucket/cp524/hf_cache"
export TRITON_CACHE_DIR="/vol/bitbucket/cp524/triton_cache"

set -ex

python mdlm_to_eval_format.py --glob_expression "../outputs/*/*/*/*/sample_evaluation/*/text_samples.jsonl" --expected_per=1

for path in ../outputs/*/*/*/fk_steering/sample_evaluation/*/*_gen.jsonl
do
    echo $path
    fname=$(basename $path)
    echo $fname
    python evaluate.py \
    --generations_file $path \
    --metrics ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext \
    --output_file "${fname}_eval.txt"
done
