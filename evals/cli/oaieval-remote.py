"""
Modal-based remote execution of evals with GPU support.
"""
import argparse
import sys
from typing import Optional, cast

import modal

# Create a Modal stub for our application
stub = modal.Stub("oaieval-remote")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evals remotely through Modal")
    parser.add_argument(
        "completion_fn",
        type=str,
        help="One or more CompletionFn URLs, separated by commas (,)",
    )
    parser.add_argument("eval", type=str, help="Name of an eval. See registry.")
    parser.add_argument("--commit_hash", type=str, help="Git commit hash to use for the eval code")
    parser.add_argument("--gpu", type=str, default="T4", help="GPU type to use (T4, A10G, A100)")
    # Add all the original oaieval arguments
    parser.add_argument("--extra_eval_params", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument("--user", type=str, default="")
    parser.add_argument("--record_path", type=str, default=None)
    parser.add_argument("--registry_path", type=str, default=None, action="append")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    return parser

# Define the Modal image that will be used for the container
@stub.function(
    gpu=modal.gpu.T4(),
    timeout=3600,
    container_idle_timeout=60,
)
def setup_and_run_eval(args_dict: dict) -> None:
    import os
    import subprocess
    
    # Clone the repository and checkout specific commit
    subprocess.run(["git", "clone", "https://github.com/fixie-ai/evals.git", "/root/evals"], check=True)
    if args_dict.get("commit_hash"):
        subprocess.run(["git", "checkout", args_dict["commit_hash"]], cwd="/root/evals", check=True)
    
    # Install the package
    subprocess.run(["pip", "install", "-e", "."], cwd="/root/evals", check=True)
    
    # Prepare the command
    cmd = ["python", "-m", "evals.cli.oaieval"]
    cmd.extend([args_dict["completion_fn"], args_dict["eval"]])
    
    # Add optional arguments
    for key, value in args_dict.items():
        if key not in ["completion_fn", "eval", "commit_hash", "gpu"] and value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    # Run the eval
    subprocess.run(cmd, check=True)

def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Configure GPU based on argument
    gpu_map = {
        "T4": modal.gpu.T4(),
        "A10G": modal.gpu.A10G(),
        "A100": modal.gpu.A100(),
    }
    setup_and_run_eval.gpu = gpu_map.get(args.gpu, modal.gpu.T4())
    
    # Run the function
    with stub.run():
        setup_and_run_eval.remote(args_dict)

if __name__ == "__main__":
    main()
