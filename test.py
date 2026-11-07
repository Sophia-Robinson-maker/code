"""Frozen-checkpoint missing-rate evaluation; exports masks and modality weights."""

import argparse
import json
from pathlib import Path

from datasets import build_dataloader
from train import configure_runtime, resolve_device, seed_everything
from utils.checkpoint import load_for_inference
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from utils.metrics import evaluator_from_loaders
from utils.options import get_args
from utils.provenance import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument(
        "--missing_rates", nargs="+", type=float, default=[0.0, 0.2, 0.3, 0.5]
    )
    p.add_argument("--gallery_missing_rate", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--gallery_file", default=None)
    p.add_argument("--query_tasks", nargs="*", default=[])
    p.add_argument(
        "--log_period",
        type=int,
        default=50,
        help="Report feature extraction and ranking progress every N batches/chunks",
    )
    cli = p.parse_args()
    if cli.log_period < 1:
        p.error("log_period must be positive")
    args = get_args([])
    vars(args).update(dict(load_train_configs(cli.config_file)))
    args.training, args.device, args.eval_seed = False, cli.device, cli.eval_seed
    args.eval_gallery_missing_rate = cli.gallery_missing_rate
    args.log_period = cli.log_period
    configure_runtime(args)
    logger = setup_logger("ORBench", save_dir=args.output_dir, if_train=False)
    if cli.gallery_file:
        args.gallery_file = cli.gallery_file
    checkpoint = cli.checkpoint or str(Path(args.output_dir) / "best.pth")
    seed_everything(args.seed)
    logger.info("Loading checkpoint: %s; device=%s", checkpoint, args.device)
    model = load_for_inference(args, checkpoint, resolve_device(args.device))
    logger.info("Model loaded; computing checkpoint SHA-256")
    checkpoint_hash = sha256_file(checkpoint)
    logger.info("Building evaluation loaders for %s", args.dataset_name)
    loaders = build_dataloader(args)
    if loaders[-1] != model.num_classes:
        raise ValueError("Dataset train class count differs from checkpoint")
    for rate_index, rate in enumerate(cli.missing_rates, 1):
        args.eval_missing_rate = rate
        args.protocol_dir = str(
            Path(args.output_dir)
            / "evaluation"
            / f"missing_{rate:g}_seed_{args.eval_seed}"
        )
        evaluator = evaluator_from_loaders(loaders[:-1], args)
        available_tasks = [
            task
            for task, loader in evaluator.query_loaders.items()
            if loader is not None
        ]
        if cli.query_tasks:
            unknown = set(cli.query_tasks) - set(available_tasks)
            if unknown:
                raise ValueError(
                    f"Unknown or unavailable query tasks: {sorted(unknown)}. "
                    f"Available tasks: {', '.join(available_tasks)}"
                )
            evaluator.query_loaders = {
                k: v for k, v in evaluator.query_loaders.items() if k in cli.query_tasks
            }
        selected_tasks = [
            task
            for task in available_tasks
            if not cli.query_tasks or task in cli.query_tasks
        ]
        logger.info(
            "Missing rate %g (%d/%d); gallery missing rate=%g; query tasks (%d): %s",
            rate,
            rate_index,
            len(cli.missing_rates),
            cli.gallery_missing_rate,
            len(selected_tasks),
            ", ".join(selected_tasks),
        )
        mean = evaluator.eval(model)
        output = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "config": vars(args),
            "missing_rate": rate,
            "gallery_missing_rate": args.eval_gallery_missing_rate,
            "seed": args.eval_seed,
            "mean_task_R1": mean,
            "results": evaluator.results,
        }
        path = Path(args.protocol_dir) / "metrics.json"
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info("Missing rate %g complete; results written to %s", rate, path)
        print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
