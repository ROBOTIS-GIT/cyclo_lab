# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Run BlinkMimic and SuperMimic training iterations in Isaac Lab."""

import json
from datetime import datetime


def run_training_loop(runner, args, root, task, contract, motions, backend, task_motions, training_probe):
    from cyclo_lab.manager_based.mimic import checkpoint
    from cyclo_lab.manager_based.mimic.export import export_sim2real_cfg
    from cyclo_lab.utils.wandb_models import model_upload_run, upload_checkpoint

    output = args.output or root / task / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    export_sim2real_cfg(runner, output)
    from torch.utils.tensorboard import SummaryWriter

    def save_checkpoint():
        return checkpoint.save(
            runner, output, contract, motions, backend, task=task, task_motions=task_motions
        )

    with SummaryWriter(str(output)) as writer, model_upload_run(output, task) as upload_run:
        for _ in range(runner.iteration, args.iterations):
            if training_probe is not None:
                training_probe(runner, task, output, backend, save_checkpoint)
            try:
                metrics = runner.train_iteration(args.collection_steps)
            except Exception as exc:
                (output / "failure.json").write_text(
                    json.dumps(
                        dict(task=task, backend=backend, iteration=runner.iteration, error=str(exc)), indent=2
                    )
                )
                raise
            print(f"Iteration {runner.iteration} {metrics}", flush=True)
            for key, value in metrics.items():
                writer.add_scalar(key, value, runner.iteration)
            if runner.iteration % 100 == 0 or runner.iteration == args.iterations:
                snapshot = save_checkpoint()

                if upload_run is not None and snapshot is not None:
                    upload_checkpoint(upload_run, snapshot)
                elif upload_run is not None and runner.iteration == args.iterations:
                    upload_checkpoint(upload_run, output / "last.pt")
