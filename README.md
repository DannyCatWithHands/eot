# EOT Benchmark

Comparing first- and second-order algorithms for solving **Entropic Optimal Transport (EOT)**.

## Structure

```
eot/
├── config.yaml          # single source of truth: problem parameters and solver settings
├── main.py              # run experiments → saves results/<problem>/
├── problems.py          # problem generators
├── solvers.py           # solver implementations
├── plot_results.ipynb   # load results/ → saves figures/
├── results/             # auto-populated, not committed
└── figures/             # auto-populated, not committed
```

## Quickstart

```bash
pip install -r requirements.txt

# Run an experiment (problem type is chosen at the command line)
python main.py --problem gaussian
python main.py --problem gene
python main.py --problem knowledge_distillation

# Open the notebook to generate plots
jupyter notebook plot_results.ipynb
```

## CLI options

```bash
# Skip solvers whose results are already saved
python main.py --problem gaussian --skip Sinkhorn Newton

# Run only specific solvers
python main.py --problem gaussian --only SORN KCRN

# Resume from a specific solver (skips all before it)
python main.py --problem gaussian --from SORN
```

## Configuration

All experiment parameters live in `config.yaml`. Problem-specific settings (including the subspace rank, regularization constant, and runtime cutoff) are nested under each problem type:

```yaml
problem:
  epsilon: 0.01
  gaussian:
    n: 2000
    rank: 300       # subspace dimension for sketch-based solvers
    L_H: 0.001      # regularization for GRN and SORN
    max_time: 100   # wall-clock cutoff in seconds
```

Solver entries contain only algorithm-specific hyperparameters:

```yaml
solvers:
  - name: SORN
    max_iter: 100
```

## Algorithms

| Config name     | Method                                         |
|-----------------|------------------------------------------------|
| `Sinkhorn`      | Sinkhorn–Knopp iterative scaling               |
| `Newton`        | Exact Newton (direct Hessian solve)            |
| `GRN`           | Regularised Newton (Tikhonov)                  |
| `SORN`          | **Proposed** — Randomized low rank Hessian Overestimation |
| `Newton_Sketch` | Randomized Subspace Newton                     |
| `SGN`           | Sketchy Global Newton                          |
| `KCRN`          | Krylov Cubic Regularized Newton                |
| `SSCN`          | Stochastic Subspace Cubic Newton               |
| `AGD`           | Nesterov accelerated gradient descent          |
| `LBFGS`         | Limited-memory BFGS                            |

## Problem types

| `--problem` value        | Description                                              | Extra dependency |
|--------------------------|----------------------------------------------------------|------------------|
| `gaussian`               | 1D Gaussian marginals, random cost matrix                | —                |
| `gene`                   | PBMC3k single-cell RNA-seq, L1 cost in PCA space         | `scanpy`         |
| `knowledge_distillation` | ResNet50/18 softmax marginals, L2 cost on class embeddings | `torch`        |
