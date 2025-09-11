# primary generation script
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import argparse

from datetime import datetime

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel

import sys
sys.path.append("fkd_diffusers")

from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

from fks_utils import do_eval, get_model

# load prompt data
def load_geneval_metadata(prompt_path, max_prompts=None):
    prompt_path = "prompt_files/" + prompt_path
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data




def main(args):
    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps

    if args.use_smc:
        assert args.resample_frequency > 0
        assert args.num_particles > 1

    # load prompt data
    prompt_data = load_geneval_metadata(args.prompt_path)

    # configure pipeline
    pipe = get_model(args.model_name)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # set output directory
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, cur_time)
    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        # make file sleep for a random time
        import time

        print("Sleeping for a random time")

        time.sleep(np.random.randint(1, 10))

        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_dir, cur_time)
        os.makedirs(output_dir, exist_ok=False)

    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    metrics_to_compute = args.metrics_to_compute.split("#") if args.metrics_to_compute != '' else []

    # cache metric fns
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )

    metrics_arr = {
        metric: dict(mean=0.0, max=0.0, min=0.0, std=0.0) for metric in metrics_to_compute
    }
    n_samples = 0
    average_time = 0

    for prompt_idx, item in enumerate(tqdm(prompt_data)):
        prompt = [item["prompt"]] * args.num_particles
        start_time = datetime.now()

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        # dump metadata
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)

        fkd_args = dict(
            lmbda=args.lmbda,
            num_particles=args.num_particles,
            use_smc=args.use_smc,
            adaptive_resampling=args.adaptive_resampling,
            resample_frequency=args.resample_frequency,
            time_steps=args.num_inference_steps,
            resampling_t_start=args.resample_t_start,
            resampling_t_end=args.resample_t_end,
            guidance_reward_fn=args.guidance_reward_fn,
            potential_type=args.potential_type,
            num_x0_samples=args.num_x0_samples, # phi
            use_log_impl=args.use_fkd_log_impl,
        )

        img_size = 1024 if args.model_name == "meissonic" else 512
        negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
        images = pipe(
            prompt=prompt, 
            negative_prompt=[negative_prompt]*len(prompt),
            height=img_size,
            width=img_size,
            guidance_scale=9.0,
            num_inference_steps=fkd_args["time_steps"],
            fkd_args=fkd_args,
            max_decode_batch_size=args.max_decode_batch_size,
            max_reward_batch_size=args.max_reward_batch_size,
        )
        images = images[0]
        if args.use_smc:
            end_time = datetime.now()

        results = do_eval(
            prompt=prompt, images=images, metrics_to_compute=metrics_to_compute
        )
        if not args.use_smc:
            end_time = datetime.now()
        time_taken = end_time - start_time #type: ignore

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = prompt_idx

        n_samples += 1

        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")
        
        print("Max GPU memory used:", torch.cuda.max_memory_allocated() / 1024**3, "GB")

        # sort images by reward
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]
        for metric in metrics_to_compute:
            results[metric]["result"] = [
                results[metric]["result"][i] for i in sorted_idx
            ]

        for metric in metrics_to_compute:
            metrics_arr[metric]["mean"] += results[metric]["mean"]
            metrics_arr[metric]["max"] += results[metric]["max"]
            metrics_arr[metric]["min"] += results[metric]["min"]
            metrics_arr[metric]["std"] += results[metric]["std"]

        for metric in metrics_to_compute:
            print(
                metric,
                metrics_arr[metric]["mean"] / n_samples,
                metrics_arr[metric]["max"] / n_samples,
            )

        if args.save_individual_images:
            sample_path = os.path.join(prompt_path, "samples")
            os.makedirs(sample_path, exist_ok=True)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            for image_idx, image in enumerate(images[:1]):
                image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))

        with open(os.path.join(prompt_path, "results.json"), "w") as f:
            json.dump(results, f)

        _, ax = plt.subplots(1, args.num_particles, figsize=(args.num_particles * 5, 5))
        if args.num_particles == 1:
            ax = [ax]
        for i, image in enumerate(images):
            ax[i].imshow(image) # type: ignore
            ax[i].axis("off") # type: ignore

        plt.suptitle(prompt[0])
        image_fpath = os.path.join(prompt_path, f"grid.png")
        plt.savefig(image_fpath)
        plt.close()

    # save final metrics
    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] /= n_samples
        metrics_arr[metric]["max"] /= n_samples
        metrics_arr[metric]["min"] /= n_samples
        metrics_arr[metric]["std"] /= n_samples

    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics_arr, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="geneval_outputs")
    parser.add_argument("--save_individual_images", type=bool, default=True)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--use_smc", action="store_true")
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#HumanPreference",
        help="# separated list of metrics",
    )
    parser.add_argument("--prompt_path", type=str, default="geneval_metadata.jsonl")

    parser.add_argument(
        "--model_name", type=str, default="stabilityai/stable-diffusion-2-1"
    )
    parser.add_argument("--lmbda", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=5)
    parser.add_argument("--resample_t_start", type=int, default=5)
    parser.add_argument("--resample_t_end", type=int, default=30)
    parser.add_argument("--potential_type", type=str, default="diff")
    parser.add_argument("--num_x0_samples", type=int, default=1, help="phi")
    parser.add_argument("--use_fkd_log_impl", action="store_true")
    parser.add_argument("--max_decode_batch_size", type=int, default=1000)
    parser.add_argument("--max_reward_batch_size", type=int, default=1000)

    args = parser.parse_args()

    if args.prompt_path == "geneval_metadata.jsonl":
        args.save_individual_images = True

    args.output_dir = args.prompt_path.replace(".jsonl", f"_outputs").replace(".json", f"_outputs")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
