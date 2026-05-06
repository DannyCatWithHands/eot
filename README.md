# EOT Benchmark

Benchmarking second-order optimization algorithms for **Entropic Optimal Transport (EOT)**.

## Structure

```
eot-benchmark/
├── config.yaml          # define your experiment here
├── solvers.py           # all solver implementations (read-only for users)
├── problems.py          # problem generators (read-only for users)
├── main.py              # run experiments → saves results/
├── plot_results.ipynb   # load results/ → saves figures/
├── results/             # auto-populated, not committed
└── figures/             # auto-populated, not committed
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Edit config.yaml to adjust solver hyperparameters if needed
# 2. Run the experiment, choosing a problem type
python main.py --problem gaussian
python main.py --problem gene
python main.py --problem knowledge_distillation

# 3. Open the notebook to plot
jupyter notebook plot_results.ipynb
```

## Algorithms

| Config name    | Method                                          |
|----------------|-------------------------------------------------|
| `Sinkhorn`     | Sinkhorn–Knopp iterative scaling                |
| `Newton`       | Exact Newton (direct Hessian solve)             |
| `GRN`          | Regularised Newton (Tikhonov)                   |
| `SORN`         | **Proposed** — Nyström low-rank + Woodbury      |
| `Newton_Sketch`| Sketched Newton (random sparse S)               |
| `SGN`          | Sketched Gauss-Newton with cubic step size      |
| `KCRN`         | Cubic regularisation in Krylov (Lanczos) space  |
| `SSCN`         | Sketched stochastic cubic Newton                |
| `AGD`          | Nesterov accelerated gradient descent           |
| `LBFGS`        | Limited-memory BFGS                             |

## Problem types

| Config `type`            | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| `gaussian`               | 1D Gaussian marginals, random cost matrix                    |
| `gene`                   | PBMC3k single-cell RNA-seq (requires `scanpy`)               |
| `knowledge_distillation` | Teacher/student softmax marginals (requires `torch`)         |
