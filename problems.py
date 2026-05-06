"""
Problem generators for EOT benchmarks.

Each generator returns (a, b, C):
    a, b : (n,) source and target marginals, sum to 1
    C    : (n, n) cost matrix, normalised to [0, 1]

Usage (via main.py, not directly):
    from problems import build_problem
    a, b, C = build_problem(config["problem"])
"""

import numpy as np


def build_problem(cfg: dict):
    """
    Dispatch to the right generator based on cfg["type"].
    Returns (a, b, C).
    """
    ptype = cfg["type"]
    _meta = {"rank", "L_H", "max_time"}
    def params(key):
        return {k: v for k, v in cfg[key].items() if k not in _meta}
    if ptype == "gaussian":
        return generate_gaussian(**params("gaussian"))
    elif ptype == "gene":
        return generate_gene(**params("gene"))
    elif ptype == "knowledge_distillation":
        return generate_knowledge_distillation(**params("knowledge_distillation"))
    else:
        raise ValueError(f"Unknown problem type: {ptype!r}")


def generate_gaussian(n, mean_a, mean_b, std_a, std_b, seed):
    """1D Gaussian marginals on a uniform grid with a random cost matrix."""
    np.random.seed(seed)

    xs = np.linspace(0, 1, n)
    C = np.random.rand(n, n)

    a = np.exp(-0.5 * ((xs - mean_a) / std_a) ** 2) + 1e-10
    b = np.exp(-0.5 * ((xs - mean_b) / std_b) ** 2) + 1e-10

    a /= a.sum()
    b /= b.sum()

    return a, b, C


def generate_gene(min_cells, n_pca_comps, seed):
    """
    Two single cells from PBMC3k as marginals.
    Cost = pairwise L1 distance between genes in PCA space, normalised to [0,1].
    Requires: scanpy
    """
    import scanpy as sc
    from scipy.spatial.distance import cdist

    np.random.seed(seed)

    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.log1p(adata)

    idx_a, idx_b = np.random.choice(adata.n_obs, size=2, replace=False)
    a = adata.X[idx_a, :].toarray().flatten()
    b = adata.X[idx_b, :].toarray().flatten()

    print(f"Sparsity of a: {np.mean(a == 0):.4f}")
    print(f"Sparsity of b: {np.mean(b == 0):.4f}")

    a /= a.sum()
    b /= b.sum()

    gene_adata = adata.T
    sc.pp.pca(gene_adata, n_comps=n_pca_comps)
    Z = gene_adata.obsm["X_pca"]

    C = cdist(Z, Z, metric="cityblock")
    C /= C.max()

    return a, b, C


def generate_knowledge_distillation(teacher, student, image, temperature):
    """
    Teacher and student softmax outputs as marginals.
    Cost = pairwise L2 distance between teacher's final-layer class embeddings,
    normalised to [0, 1].
    Requires: torch, torchvision, Pillow
    """
    import torch
    import torch.nn.functional as F
    import torchvision.models as models
    from torchvision import transforms
    from PIL import Image

    model_map = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    }

    teacher_fn, teacher_weights = model_map[teacher]
    student_fn, student_weights = model_map[student]

    teacher_model = teacher_fn(weights=teacher_weights).eval()
    student_model = student_fn(weights=student_weights).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image.open(image)).unsqueeze(0)

    with torch.no_grad():
        a = F.softmax(teacher_model(input_tensor).squeeze() / temperature, dim=0).numpy()
        b = F.softmax(student_model(input_tensor).squeeze() / temperature, dim=0).numpy()

        class_embeddings = teacher_model.fc.weight
        C = torch.cdist(class_embeddings, class_embeddings, p=2.0)
        C = (C / C.max()).numpy()

    return a, b, C
