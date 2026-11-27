# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Kiwoong Park

"""Run BlinkMimic and SuperMimic training iterations in Isaac Lab."""

import json
import time
from collections import defaultdict
from datetime import datetime


class TrainingLogWindow:
    """Average sampled metrics, retaining counters and schedules at their latest value."""

    LATEST = {
        "Perf/total_steps", "Perf/wall_hours",
        "Loss/learning_rate", "Loss/policy_learning_rate",
        "Policy/learning_rate_multiplier", "BlinkMimic/nonfinite_updates",
        "BlinkMimic/normalized_episode_return", "BlinkMimic/termination_fraction",
    }

    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.latest = {}
        self.iterations = 0
        self.seconds = 0.0
        self.steps = 0

    def add(self, metrics, *, seconds, steps):
        self.iterations += 1
        self.seconds += seconds
        self.steps += steps
        for key, value in metrics.items():
            if key in self.LATEST:
                self.latest[key] = value
            else:
                self.sums[key] += value
                self.counts[key] += 1

    def scalars(self):
        return {key: value / self.counts[key] for key, value in self.sums.items()} | self.latest | {
            "Perf/iteration_time": self.seconds / self.iterations,
            "Perf/iterations_per_second": self.iterations / max(self.seconds, 1e-9),
            "Perf/total_fps": self.steps / max(self.seconds, 1e-9),
        }


def run_training_loop(runner, args, root, task, motions, backend, task_motions, training_probe):
    from isaaclab_rl.entrypoints.common import dump_train_configs

    from cyclo_lab.manager_based.mimic._shared import checkpoint
    from cyclo_lab.manager_based.mimic._shared.export import export_sim2real_cfg
    from cyclo_lab.utils.wandb_models import model_upload_run, upload_checkpoint

    is_supermimic = runner.family == "supermimic"
    log_interval = runner.cfg.log_interval if is_supermimic else runner.cfg.log_every
    if log_interval < 1:
        raise ValueError("Logging interval must be positive")
    output = args.output or root / task / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    env_cfg = runner.env.unwrapped.cfg
    dump_train_configs(str(output), env_cfg, runner.cfg)
    export_sim2real_cfg(runner, output)
    from cyclo_lab.utils.export_git_state import export_git_state

    export_git_state(output, __file__)
    from torch.utils.tensorboard import SummaryWriter

    def save_checkpoint():
        return checkpoint.save(
            runner, output, motions, task=task, task_motions=task_motions,
        )

    with SummaryWriter(str(output)) as writer, model_upload_run(output, task) as upload_run:
        previous_wall_time = runner.metrics.wall_time
        start_time = time.perf_counter()
        window = TrainingLogWindow()
        for _ in range(runner.iteration, args.iterations):
            if training_probe is not None:
                training_probe(runner, task, output, backend, save_checkpoint, motions)
            iteration_start = time.perf_counter()
            previous_steps = runner.metrics.total_steps
            try:
                metrics = runner.train_iteration(args.collection_steps)
            except Exception as exc:
                (output / "failure.json").write_text(
                    json.dumps(
                        dict(task=task, backend=backend, iteration=runner.iteration, error=str(exc)), indent=2
                    )
                )
                raise
            iteration_time = time.perf_counter() - iteration_start
            runner.metrics.wall_time = previous_wall_time + time.perf_counter() - start_time
            metrics.update({
                "Perf/total_steps": runner.metrics.total_steps,
                "Perf/wall_hours": runner.metrics.wall_time / 3600,
            })
            window.add(metrics, seconds=iteration_time, steps=runner.metrics.total_steps - previous_steps)
            # SuperMimic labels the first completed step as zero; BlinkMimic
            # labels completed iterations starting at one.
            log_step = runner.iteration - 1 if is_supermimic else runner.iteration
            if log_step % log_interval == 0 or (not is_supermimic and runner.iteration == args.iterations):
                logged = window.scalars()
                display_step = log_step // log_interval if is_supermimic else log_step
                display_total = args.iterations // log_interval if is_supermimic else args.iterations
                heading = f" Learning iteration {display_step}/{display_total} "
                lines = [
                    f"\n{heading:=^80}",
                    f"{'Iteration time (s/iter)':>36}: {logged['Perf/iteration_time']:.2f}",
                    *(f"{key:>36}: {value:.6g}" for key, value in logged.items() if key != "Perf/iteration_time"),
                ]
                print("\n".join(lines), flush=True)
                for key, value in logged.items():
                    writer.add_scalar(key, value, log_step)
                runner.metrics.reset_interval()
                window = TrainingLogWindow()
            if log_step % 100 == 0 or runner.iteration == args.iterations:
                snapshot = save_checkpoint()

                if upload_run is not None and snapshot is not None:
                    upload_checkpoint(upload_run, snapshot)
                elif upload_run is not None and runner.iteration == args.iterations:
                    upload_checkpoint(upload_run, output / "last.pt")
