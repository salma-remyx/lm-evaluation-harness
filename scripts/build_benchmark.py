import argparse
import json
import logging
import os

import yaml


try:
    from scripts.chain_of_problems import build_chain_benchmark
except ModuleNotFoundError:  # allow running as a standalone script
    from chain_of_problems import build_chain_benchmark


# from lm_eval.api.registry import ALL_TASKS
eval_logger = logging.getLogger(__name__)


# from lm_eval.tasks import include_task_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name")
    parser.add_argument("--benchmark_path")
    parser.add_argument("--task_save_path", default="lm_eval/tasks/")
    parser.add_argument(
        "--chain_of_problems",
        action="store_true",
        help="Build a Scheherazade-style chain-of-problems benchmark JSONL.",
    )
    parser.add_argument("--seed_path", help="JSON seed of problem chains.")
    parser.add_argument("--output_path", default="chain_of_problems.jsonl")
    parser.add_argument(
        "--direction", default="forward", choices=["forward", "backward"]
    )
    parser.add_argument("--chain_length", type=int, default=0)
    return parser.parse_args()


def build_chained_benchmark(seed_path, output_path, direction, chain_length):
    """Compose a chain-of-problems benchmark and write it as GSM8K-style JSONL."""
    with open(seed_path, encoding="utf-8") as file:
        seed = json.load(file)
    records = build_chain_benchmark(
        seed, direction=direction, chain_length=chain_length
    )
    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines(json.dumps(record) + "\n" for record in records)
    eval_logger.info(f"Wrote {len(records)} chained problems to {output_path}")
    return records


if __name__ == "__main__":
    args = parse_args()

    if args.chain_of_problems:
        build_chained_benchmark(
            args.seed_path, args.output_path, args.direction, args.chain_length
        )
        raise SystemExit(0)

    from promptsource.templates import DatasetTemplates
    from tqdm import tqdm

    with open(args.benchmark_path, encoding="utf-8") as file:
        TASK_LIST = yaml.full_load(file)
        for task in tqdm(TASK_LIST):
            eval_logger.info(f"Processing {task}")

            dataset_name = task["dataset_path"]
            if "dataset_name" in task:
                subset_name = task["dataset_name"]
                file_subdir = f"{dataset_name}/{subset_name}"
            else:
                subset_name = None
                file_subdir = f"{dataset_name}"

            file_path = os.path.join(args.task_save_path, file_subdir, "promptsource/")

            os.makedirs(file_path, exist_ok=True)

            if subset_name is None:
                prompts = DatasetTemplates(dataset_name=dataset_name)
            else:
                prompts = DatasetTemplates(
                    dataset_name=dataset_name, subset_name=subset_name
                )

            for idx, prompt_name in enumerate(prompts.all_template_names):
                full_file_name = f"promptsource_{idx}.yaml"
                config_dict = {
                    "group": args.benchmark_name,
                    "include": "promptsource_template.yaml",
                    "use_prompts": f"promptsource:{prompt_name}",
                }

                file_save_path = os.path.join(file_path, full_file_name)
                eval_logger.info(f"Save to {file_save_path}")
                with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                    yaml.dump(config_dict, yaml_file)
