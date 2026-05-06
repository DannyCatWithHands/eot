"""
Experiment runner.

Usage:
    python main.py --problem gaussian
    python main.py --problem gaussian --skip Sinkhorn Newton
    python main.py --problem gaussian --only SORN KCRN
    python main.py --problem gaussian --config config.yaml  # explicit config path

For each solver listed in the config, runs it on the chosen problem and saves
metrics to results/{solver}_{problem}_eps{epsilon}_{params}.pkl
"""

import argparse
import os
import pickle
import yaml

from problems import build_problem
from solvers import (
    sinkhorn_algorithm,
    newton_method,
    ron_method,
    newton_sketch_method,
    sgn_method,
    kcrn_method,
    sscn_method,
    agd_method,
    lbfgs_method,
)

# (function, kwarg_keys_from_config, rank_param_or_None, L_H_param_or_None)
SOLVER_REGISTRY = {
    "Sinkhorn":      (sinkhorn_algorithm,    ["max_iter"],        None,           None),
    "Newton":        (newton_method,         ["max_iter", "reg"], None,           None),
    "GRN":           (newton_method,         ["max_iter", "reg"], None,           "L_H"),
    "SORN":          (ron_method,            ["max_iter"],        "k",            "L_H"),
    "Newton_Sketch": (newton_sketch_method,  ["max_iter"],        "sketch_dim",   None),
    "SGN":           (sgn_method,            ["max_iter", "L"],   "sketch_dim",   None),
    "KCRN":          (kcrn_method,           ["max_iter"],        "lanczos_iter", None),
    "SSCN":          (sscn_method,           ["max_iter"],        "sketch_dim",   None),
    "AGD":           (agd_method,            ["max_iter"],        None,           None),
    "LBFGS":         (lbfgs_method,          ["max_iter"],        None,           None),
}


def build_filename(solver_cfg, problem_cfg, results_dir):
    """Build a deterministic result filename from config values."""
    name = solver_cfg["name"]
    ptype = problem_cfg["type"]
    eps = problem_cfg["epsilon"]
    params = {k: v for k, v in solver_cfg.items() if k != "name"}
    param_str = "_".join(f"{k}{v}" for k, v in params.items())
    return os.path.join(results_dir, f"{name}_{ptype}_eps{eps}_{param_str}.pkl")


def run_experiment(cfg):
    ptype = cfg["problem"]["type"]
    results_dir = os.path.join(cfg["output"]["results_dir"], ptype)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cfg["output"]["figures_dir"], exist_ok=True)
    print(f"Building problem: {ptype}")
    a, b, C = build_problem(cfg["problem"])
    epsilon = cfg["problem"]["epsilon"]
    problem_sub = cfg["problem"][ptype]
    rank = problem_sub["rank"]
    L_H  = problem_sub["L_H"]

    for solver_cfg in cfg["solvers"]:
        name = solver_cfg["name"]
        if name not in SOLVER_REGISTRY:
            print(f"  [SKIP] Unknown solver: {name}")
            continue

        func, kwarg_keys, rank_param, L_H_param = SOLVER_REGISTRY[name]
        kwargs = {k: solver_cfg[k] for k in kwarg_keys if k in solver_cfg}
        if rank_param is not None:
            kwargs[rank_param] = rank
        if L_H_param is not None:
            kwargs[L_H_param] = L_H
        kwargs["max_time"] = problem_sub["max_time"]

        print(f"\n--- Running {name} ---")
        _, metrics, comp_time, total_iter = func(a, b, C, epsilon, **kwargs)

        final_error = metrics["error"][-1] if metrics.get("error") else None
        print(f"  Time: {comp_time:.4f}s | Iters: {total_iter} | Final grad norm: {final_error:.2e}")

        out_path = build_filename(solver_cfg, cfg["problem"], results_dir)
        with open(out_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"  Saved → {out_path}")


VALID_PROBLEMS = ["gaussian", "gene", "knowledge_distillation"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, choices=VALID_PROBLEMS,
                        help="Problem type to solve")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Solver names to skip")
    parser.add_argument("--only", nargs="+", default=[],
                        help="Run only these solvers (overrides --skip)")
    parser.add_argument("--from", dest="start_from", default=None,
                        help="Start from this solver, skipping all before it")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["problem"]["type"] = args.problem

    if args.only:
        cfg["solvers"] = [s for s in cfg["solvers"] if s["name"] in args.only]
    else:
        if args.start_from:
            names = [s["name"] for s in cfg["solvers"]]
            if args.start_from not in names:
                raise ValueError(f"Unknown solver: {args.start_from!r}. Valid names: {names}")
            idx = names.index(args.start_from)
            cfg["solvers"] = cfg["solvers"][idx:]
        if args.skip:
            cfg["solvers"] = [s for s in cfg["solvers"] if s["name"] not in args.skip]

    run_experiment(cfg)
