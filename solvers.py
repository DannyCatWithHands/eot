"""
EOT solvers — pure numerical code, no I/O or plotting.

Every solver has the same signature:
    solver(a, b, C, epsilon, **kwargs)
    -> P, metrics, computation_time, iterations

where:
    a, b        : (n,) marginals
    C           : (n, n) cost matrix
    epsilon     : regularisation parameter
    P           : (n, n) optimal transport plan
    metrics     : dict of lists, one value per iteration
    computation_time : float (seconds)
    iterations  : int
"""

import numpy as np
import time
from scipy.optimize import minimize
from tqdm import tqdm


# ─── Shared helpers ───────────────────────────────────────────────────────────

def compute_metrics(P, a, b, C, epsilon):
    cost = np.sum(P * C)
    entropy = -np.sum(P * np.log(P + 1e-16))
    primal_obj = cost - epsilon * entropy
    marginal_a_violation = np.linalg.norm(P.sum(axis=1) - a, ord=1)
    marginal_b_violation = np.linalg.norm(P.sum(axis=0) - b, ord=1)
    total_marginal_violation = marginal_a_violation + marginal_b_violation
    return {
        'cost': cost,
        'entropy': entropy,
        'primal_obj': primal_obj,
        'marginal_a_violation': marginal_a_violation,
        'marginal_b_violation': marginal_b_violation,
        'total_marginal_violation': total_marginal_violation
    }

def newton_objective(dual_vars, a, b, C, epsilon):
    n = len(a)
    if not isinstance(dual_vars, np.ndarray):
        dual_vars = np.zeros(2*n, dtype=float)
    f_vals = dual_vars[:n].reshape(n)
    g_vals = dual_vars[n:].reshape(n)
    f_col = f_vals.reshape(-1, 1)
    g_row = g_vals.reshape(1, -1)
    K = np.exp((f_col + g_row - C) / epsilon)
    obj = np.dot(f_vals, a) + np.dot(g_vals, b) - epsilon * np.sum(K)
    return -obj

def newton_gradient(dual_vars, a, b, C, epsilon):
    n = len(a)
    if not isinstance(dual_vars, np.ndarray):
        dual_vars = np.zeros(2*n, dtype=float)
    f_vals = dual_vars[:n].reshape(n)
    g_vals = dual_vars[n:].reshape(n)
    f_col = f_vals.reshape(-1, 1)
    g_row = g_vals.reshape(1, -1)
    K = np.exp((f_col + g_row - C) / epsilon)
    a_current = np.sum(K, axis=1)
    b_current = np.sum(K, axis=0)
    grad = np.concatenate([a - a_current, b - b_current])
    return -grad

def newton_hessian(dual_vars, a, b, C, epsilon):
    n = len(a)
    if not isinstance(dual_vars, np.ndarray):
        dual_vars = np.zeros(2*n, dtype=float)
    f_vals = dual_vars[:n].reshape(n)
    g_vals = dual_vars[n:].reshape(n)
    f_col = f_vals.reshape(-1, 1)
    g_row = g_vals.reshape(1, -1)
    K = np.exp((f_col + g_row - C) / epsilon)
    H = np.zeros((2*n, 2*n))
    for i in range(n):
        H[i, i] = np.sum(K[i, :]) / epsilon
        H[n+i, n+i] = np.sum(K[:, i]) / epsilon
    for i in range(n):
        for j in range(n):
            H[i, n+j] = K[i, j] / epsilon
            H[n+j, i] = K[i, j] / epsilon
    return H

def generate_S(d, s):
    matrix = np.zeros((d, s), dtype=int)
    col_indices = np.random.randint(0, s, size=d)
    matrix[np.arange(d), col_indices] = 1
    return matrix

def lanczos(A, g, k):
    n = A.shape[0]
    Q = np.zeros((n, k + 1))
    T = np.zeros((k, k))
    q = g.copy()
    q /= np.linalg.norm(g)
    b = np.zeros(k)
    b[0] = 1
    b = np.linalg.norm(g) * b
    Q[:, 0] = q
    beta = 0
    for j in range(k):
        z = A @ Q[:, j]
        alpha = np.dot(Q[:, j], z)
        if j > 0:
            z -= beta * Q[:, j - 1]
        z -= alpha * Q[:, j]
        beta = np.linalg.norm(z)
        T[j, j] = alpha
        if j < k - 1:
            T[j, j + 1] = beta
            T[j + 1, j] = beta
            Q[:, j + 1] = z / beta
    return Q[:, :k], b, T

def cubic_model(p, g, H, sigma):
    return np.dot(p, g) + 0.5 * np.dot(p, H @ p) + (sigma / 6) * np.linalg.norm(p)**3

def grad_cubic_model(p, g, H, sigma):
    norm_p = np.linalg.norm(p)
    if norm_p == 0:
        cubic_grad = np.zeros_like(p)
    else:
        cubic_grad = sigma * norm_p * p
    return g + H @ p + cubic_grad / 2


# ─── Solvers ──────────────────────────────────────────────────────────────────

def sinkhorn_algorithm(a, b, C, epsilon, max_iter=300, tol=1e-8, warm_start_iter=0, max_time=None):
    max_iter = max_iter + warm_start_iter
    n = len(a)
    K = np.exp(-C / epsilon)
    u = np.ones(n)
    v = np.ones(n)
    metrics = {
        'iterations': [], 'error': [], 'cost': [], 'entropy': [],
        'primal_obj': [], 'dual_obj': [], 'marginal_a_violation': [],
        'marginal_b_violation': [], 'total_marginal_violation': [],
        'scaling_diff': [], 'time': [0.0]
    }
    for i in tqdm(range(max_iter)):
        start_time = time.time()
        u_prev = u.copy()
        Q = np.diag(u) @ K @ np.diag(v)
        grad = np.concatenate([a - np.sum(Q, axis=1), b - np.sum(Q, axis=0)])
        grad_norm = np.linalg.norm(grad)
        v = b / (K.T @ u)
        u = a / (K @ v)
        P = np.diag(u) @ K @ np.diag(v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_obj = np.sum(f * a) + np.sum(g * b) - epsilon * np.sum(u * K @ v)
        current_metrics = compute_metrics(P, a, b, C, epsilon)
        end_time = time.time()
        if i >= warm_start_iter:
            metrics['iterations'].append(i)
            if i % 10 == 0: print(grad_norm)
            metrics['error'].append(grad_norm)
            metrics['cost'].append(current_metrics['cost'])
            metrics['entropy'].append(current_metrics['entropy'])
            metrics['primal_obj'].append(current_metrics['primal_obj'])
            metrics['dual_obj'].append(dual_obj)
            metrics['marginal_a_violation'].append(current_metrics['marginal_a_violation'])
            metrics['marginal_b_violation'].append(current_metrics['marginal_b_violation'])
            metrics['total_marginal_violation'].append(current_metrics['total_marginal_violation'])
            metrics['scaling_diff'].append(np.linalg.norm(u - u_prev))
            metrics['time'].append(metrics['time'][-1] + end_time - start_time)
            if max_time is not None and metrics['time'][-1] >= max_time:
                break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    computation_time = 0
    for t in range(len(metrics['time'])):
        computation_time += metrics['time'][t]
    iterations = i + 1
    metrics['time'].insert(0, 0)
    metrics['time'].pop()
    return P, metrics, computation_time, iterations


def newton_method(a, b, C, epsilon, L_H=0.005, tol=1e-8, max_iter=50,
                  warm_start_iter=0, reg=False, max_time=None):
    n = len(a)
    dual_vars = np.zeros(2*n)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for i in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_vars = np.concatenate([f, g])
    metrics = {
        'iterations': [], 'error': [], 'cost': [], 'entropy': [],
        'primal_obj': [], 'dual_obj': [], 'marginal_a_violation': [],
        'marginal_b_violation': [], 'total_marginal_violation': [],
        'gradient_norm': [], 'time': [0.0]
    }
    iteration = 0
    for iteration in tqdm(range(max_iter)):
        start_time = time.time()
        grad = newton_gradient(dual_vars, a, b, C, epsilon)
        grad_norm = np.linalg.norm(grad)
        hess = newton_hessian(dual_vars, a, b, C, epsilon)
        if reg:
            hess += (L_H * grad_norm**1/2) * np.eye(2*n)
        try:
            newton_dir = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            newton_dir = np.linalg.solve(hess + 1e-12*np.eye(2*n), -grad)
        alpha = 1.0
        c = 0.5
        max_backtrack = 100
        current_obj = newton_objective(dual_vars, a, b, C, epsilon)
        for _ in range(max_backtrack):
            new_dual_vars = dual_vars + alpha * newton_dir
            new_obj = newton_objective(new_dual_vars, a, b, C, epsilon)
            if new_obj < current_obj:
                dual_vars = new_dual_vars
                break
            alpha *= c
        else:
            if iteration < 10:
                print('line search failed in iter', iteration)
            dual_vars += 1e-4 * newton_dir
        f = dual_vars[:n]
        g = dual_vars[n:]
        K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
        K_sum = K.sum()
        if K_sum > 0:
            P = K / K_sum
        else:
            P = np.zeros((n, n))
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        current_metrics = compute_metrics(P, a, b, C, epsilon)
        end_time = time.time()
        metrics['iterations'].append(iteration)
        if iteration % 10 == 0: print(grad_norm)
        metrics['error'].append(grad_norm)
        metrics['cost'].append(current_metrics['cost'])
        metrics['entropy'].append(current_metrics['entropy'])
        metrics['primal_obj'].append(current_metrics['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(current_metrics['marginal_a_violation'])
        metrics['marginal_b_violation'].append(current_metrics['marginal_b_violation'])
        metrics['total_marginal_violation'].append(current_metrics['total_marginal_violation'])
        metrics['gradient_norm'].append(grad_norm)
        metrics['time'].append(metrics['time'][-1] + (end_time - start_time))
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    computation_time = 0
    for t in range(len(metrics['time'])):
        computation_time += metrics['time'][t]
    return P, metrics, computation_time, metrics['iterations'][-1]


def ron_method(a, b, C, epsilon, L_H=0.01, k=20, max_iter=50, tol=1e-8,
               warm_start_iter=0, max_time=None):
    """SORN: low-rank Nyström Hessian approximation via column sampling."""
    n = len(a)
    dim = 2 * n
    dual_vars = np.zeros(dim, dtype=float)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for i in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_vars = np.concatenate([f, g])
    metrics = {
        'iterations': [], 'error': [], 'cost': [], 'entropy': [],
        'primal_obj': [], 'dual_obj': [], 'marginal_a_violation': [],
        'marginal_b_violation': [], 'total_marginal_violation': [],
        'gradient_norm': [], 'step_size': [], 'time': [0.0]
    }
    for i in tqdm(range(max_iter)):
        start_time = time.time()
        grad = newton_gradient(dual_vars, a, b, C, epsilon)
        grad_norm = np.linalg.norm(grad)
        hess = newton_hessian(dual_vars, a, b, C, epsilon)
        diag = np.diag(hess).copy()
        F = np.zeros((dim, k))
        for j in range(k):
            prob = np.nan_to_num(diag)
            prob[prob < 0] = 0
            sum_prob = np.sum(prob)
            if sum_prob < 1e-13:
                F = F[:, :j]
                break
            prob = prob / sum_prob
            try:
                s = np.random.choice(dim, p=prob)
            except ValueError:
                s = np.argmax(diag)
            col_s = hess[:, s]
            if j > 0:
                col_s = col_s - (F[:, 0:j] @ F[s, 0:j].T).reshape(-1)
            diag = np.maximum(0, np.nan_to_num(diag - (col_s**2) / col_s[s]))
            F[:, j] = col_s / np.sqrt(col_s[s])
        if F.shape[1] > 0:
            delta = (L_H * np.linalg.norm(grad))**1/2
            D_inv = 1.0 / (diag + delta)
            D_inv_F = D_inv.reshape(-1, 1) * F
            inner_term = np.eye(F.shape[1]) + F.T @ D_inv_F
            inner_inv = np.linalg.inv(inner_term)
            direction = -D_inv * grad + D_inv_F @ inner_inv @ (F.T @ (D_inv * grad))
        else:
            if i < 10:
                print('used gd in iter', i)
            direction = -grad / (diag + 1e-10)
        alpha = 1.0
        sufficient_decrease = 0.1
        backtracking_factor = 0.5
        current_obj = newton_objective(dual_vars, a, b, C, epsilon)
        for _ in range(100):
            new_vars = dual_vars + alpha * direction
            new_obj = newton_objective(new_vars, a, b, C, epsilon)
            if new_obj <= current_obj + sufficient_decrease * alpha * grad.dot(direction):
                break
            alpha *= backtracking_factor
        dual_vars = dual_vars + alpha * direction
        f_vals = dual_vars[:n]
        g_vals = dual_vars[n:]
        K = np.exp((f_vals.reshape(-1, 1) + g_vals.reshape(1, -1) - C) / epsilon)
        K_sum = np.sum(K)
        if K_sum > 0:
            P = K / K_sum
        else:
            P = np.zeros((n, n))
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        current_metrics = compute_metrics(P, a, b, C, epsilon)
        end_time = time.time()
        metrics['iterations'].append(i)
        if i % 10 == 0: print(grad_norm)
        metrics['error'].append(grad_norm)
        metrics['cost'].append(current_metrics['cost'])
        metrics['entropy'].append(current_metrics['entropy'])
        metrics['primal_obj'].append(current_metrics['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(current_metrics['marginal_a_violation'])
        metrics['marginal_b_violation'].append(current_metrics['marginal_b_violation'])
        metrics['total_marginal_violation'].append(current_metrics['total_marginal_violation'])
        metrics['gradient_norm'].append(grad_norm)
        metrics['step_size'].append(alpha)
        metrics['time'].append(metrics['time'][-1] + (end_time - start_time))
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    computation_time = 0
    for t in range(len(metrics['time'])):
        computation_time += metrics['time'][t]
    iterations = i + 1
    f_vals = dual_vars[:n]
    g_vals = dual_vars[n:]
    K = np.exp((f_vals.reshape(-1, 1) + g_vals.reshape(1, -1) - C) / epsilon)
    P = K / np.sum(K)
    metrics['time'].insert(0, 0)
    metrics['time'].pop()
    return P, metrics, computation_time, iterations


def newton_sketch_method(a, b, C, epsilon, sketch_dim=150, tol=1e-8,
                         max_iter=50, warm_start_iter=0, max_time=None):
    n = len(a)
    dual_vars = np.zeros(2*n)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for i in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_vars = np.concatenate([f, g])
    metrics = {
        'iterations': [], 'error': [], 'cost': [], 'entropy': [],
        'primal_obj': [], 'dual_obj': [], 'marginal_a_violation': [],
        'marginal_b_violation': [], 'total_marginal_violation': [],
        'gradient_norm': [], 'time': [0.0]
    }
    iteration = 0
    for iteration in tqdm(range(max_iter)):
        start_time = time.time()
        grad = newton_gradient(dual_vars, a, b, C, epsilon)
        grad_norm = np.linalg.norm(grad)
        hess = newton_hessian(dual_vars, a, b, C, epsilon)
        S = generate_S(2*n, sketch_dim)
        STHS = S.T @ hess @ S
        try:
            interm = np.linalg.solve(STHS, -S.T @ grad)
        except np.linalg.LinAlgError:
            interm = np.linalg.lstsq(STHS, -S.T @ grad, rcond=None)[0]
        newton_dir = S @ interm
        alpha = 1.0
        c = 0.5
        max_backtrack = 100
        current_obj = newton_objective(dual_vars, a, b, C, epsilon)
        for _ in range(max_backtrack):
            new_dual_vars = dual_vars + alpha * newton_dir
            new_obj = newton_objective(new_dual_vars, a, b, C, epsilon)
            if new_obj < current_obj:
                dual_vars = new_dual_vars
                break
            alpha *= c
        else:
            dual_vars += 1e-4 * newton_dir
        f = dual_vars[:n]
        g = dual_vars[n:]
        K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
        K_sum = K.sum()
        if K_sum > 0:
            P = K / K_sum
        else:
            P = np.zeros((n, n))
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        current_metrics = compute_metrics(P, a, b, C, epsilon)
        end_time = time.time()
        metrics['iterations'].append(iteration)
        if iteration % 10 == 0: print(grad_norm)
        metrics['error'].append(grad_norm)
        metrics['cost'].append(current_metrics['cost'])
        metrics['entropy'].append(current_metrics['entropy'])
        metrics['primal_obj'].append(current_metrics['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(current_metrics['marginal_a_violation'])
        metrics['marginal_b_violation'].append(current_metrics['marginal_b_violation'])
        metrics['total_marginal_violation'].append(current_metrics['total_marginal_violation'])
        metrics['gradient_norm'].append(grad_norm)
        metrics['time'].append(metrics['time'][-1] + (end_time - start_time))
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    computation_time = 0
    for t in range(len(metrics['time'])):
        computation_time += metrics['time'][t]
    iterations = iteration + 1
    f = dual_vars[:n]
    g = dual_vars[n:]
    K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
    P = K / K.sum()
    metrics['time'].insert(0, 0)
    metrics['time'].pop()
    return P, metrics, computation_time, iterations


def sgn_method(a, b, C, epsilon, sketch_dim=150, L=10, tol=1e-8,
               max_iter=50, warm_start_iter=0, max_time=None):
    n = len(a)
    dual_vars = np.zeros(2*n)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for i in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_vars = np.concatenate([f, g])
    metrics = {
        'iterations': [], 'error': [], 'cost': [], 'entropy': [],
        'primal_obj': [], 'dual_obj': [], 'marginal_a_violation': [],
        'marginal_b_violation': [], 'total_marginal_violation': [],
        'gradient_norm': [], 'time': [0.0]
    }
    iteration = 0
    for iteration in tqdm(range(max_iter)):
        start_time = time.time()
        grad = newton_gradient(dual_vars, a, b, C, epsilon)
        grad_norm = np.linalg.norm(grad)
        hess = newton_hessian(dual_vars, a, b, C, epsilon)
        S = generate_S(2*n, sketch_dim)
        STHS = S.T @ hess @ S
        grad_S = S.T @ grad
        try:
            interm = np.linalg.solve(STHS, -grad_S)
        except np.linalg.LinAlgError:
            interm = np.linalg.lstsq(STHS, -grad_S, rcond=None)[0]
        local_grad_norm = np.sqrt(np.abs(np.dot(grad_S, -interm)))
        newton_dir = ((-1 + np.sqrt(1 + 2*L*local_grad_norm)) / (L*local_grad_norm)) * S @ interm
        alpha = 1.0
        c = 0.5
        max_backtrack = 100
        current_obj = newton_objective(dual_vars, a, b, C, epsilon)
        for _ in range(max_backtrack):
            new_dual_vars = dual_vars + alpha * newton_dir
            new_obj = newton_objective(new_dual_vars, a, b, C, epsilon)
            if new_obj < current_obj:
                dual_vars = new_dual_vars
                break
            alpha *= c
        else:
            dual_vars += 1e-4 * newton_dir
        f = dual_vars[:n]
        g = dual_vars[n:]
        K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
        K_sum = K.sum()
        if K_sum > 0:
            P = K / K_sum
        else:
            P = np.zeros((n, n))
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        current_metrics = compute_metrics(P, a, b, C, epsilon)
        end_time = time.time()
        metrics['iterations'].append(iteration)
        if iteration % 10 == 0: print(grad_norm)
        metrics['error'].append(grad_norm)
        metrics['cost'].append(current_metrics['cost'])
        metrics['entropy'].append(current_metrics['entropy'])
        metrics['primal_obj'].append(current_metrics['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(current_metrics['marginal_a_violation'])
        metrics['marginal_b_violation'].append(current_metrics['marginal_b_violation'])
        metrics['total_marginal_violation'].append(current_metrics['total_marginal_violation'])
        metrics['gradient_norm'].append(grad_norm)
        metrics['time'].append(metrics['time'][-1] + (end_time - start_time))
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    computation_time = 0
    for t in range(len(metrics['time'])):
        computation_time += metrics['time'][t]
    iterations = iteration + 1
    f = dual_vars[:n]
    g = dual_vars[n:]
    K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
    P = K / K.sum()
    metrics['time'].insert(0, 0)
    metrics['time'].pop()
    return P, metrics, computation_time, iterations


def kcrn_method(a, b, C, epsilon, lanczos_iter=150, M=0.01, tol=1e-8,
                max_iter=50, warm_start_iter=0, max_time=None):
    n = len(a)
    dual_vars = np.zeros(2*n)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for i in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_vars = np.concatenate([f, g])
    metrics = {
        'iterations': [], 'error': [], 'cost': [], 'entropy': [],
        'primal_obj': [], 'dual_obj': [], 'marginal_a_violation': [],
        'marginal_b_violation': [], 'total_marginal_violation': [],
        'gradient_norm': [], 'time': [0.0]
    }
    iteration = 0
    for iteration in tqdm(range(max_iter)):
        start_time = time.time()
        grad = newton_gradient(dual_vars, a, b, C, epsilon)
        grad_norm = np.linalg.norm(grad)
        hess = newton_hessian(dual_vars, a, b, C, epsilon)
        V, grad_, H = lanczos(hess, grad, lanczos_iter)
        def sub_p(z):
            return cubic_model(z, grad_, H, M)
        def sub_jac(z):
            return grad_cubic_model(z, grad_, H, M)
        newton_dir = V @ minimize(sub_p, np.zeros(lanczos_iter), method='trust-constr',
                                  jac=sub_jac, options={'maxiter': 30}).x
        alpha = 1.0
        c = 0.5
        max_backtrack = 100
        current_obj = newton_objective(dual_vars, a, b, C, epsilon)
        for _ in range(max_backtrack):
            new_dual_vars = dual_vars + alpha * newton_dir
            new_obj = newton_objective(new_dual_vars, a, b, C, epsilon)
            if new_obj < current_obj:
                dual_vars = new_dual_vars
                break
            alpha *= c
        else:
            dual_vars += 1e-4 * newton_dir
        f = dual_vars[:n]
        g = dual_vars[n:]
        K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
        K_sum = K.sum()
        if K_sum > 0:
            P = K / K_sum
        else:
            P = np.zeros((n, n))
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        current_metrics = compute_metrics(P, a, b, C, epsilon)
        end_time = time.time()
        metrics['iterations'].append(iteration)
        if iteration % 10 == 0: print(grad_norm)
        metrics['error'].append(grad_norm)
        metrics['cost'].append(current_metrics['cost'])
        metrics['entropy'].append(current_metrics['entropy'])
        metrics['primal_obj'].append(current_metrics['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(current_metrics['marginal_a_violation'])
        metrics['marginal_b_violation'].append(current_metrics['marginal_b_violation'])
        metrics['total_marginal_violation'].append(current_metrics['total_marginal_violation'])
        metrics['gradient_norm'].append(grad_norm)
        metrics['time'].append((metrics['time'][-1] if metrics['time'] else 0) + (end_time - start_time))
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    computation_time = 0
    for t in range(len(metrics['time'])):
        computation_time += metrics['time'][t]
    iterations = iteration + 1
    f = dual_vars[:n]
    g = dual_vars[n:]
    K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
    P = K / K.sum()
    metrics['time'].insert(0, 0)
    metrics['time'].pop()
    return P, metrics, computation_time, iterations


def sscn_method(a, b, C, epsilon, sketch_dim=150, M=0.01, tol=1e-8,
                max_iter=50, warm_start_iter=0, max_time=None):
    n = len(a)
    dual_vars = np.zeros(2*n)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for i in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_vars = np.concatenate([f, g])
    metrics = {
        'iterations': [], 'error': [], 'cost': [], 'entropy': [],
        'primal_obj': [], 'dual_obj': [], 'marginal_a_violation': [],
        'marginal_b_violation': [], 'total_marginal_violation': [],
        'gradient_norm': [], 'time': [0.0]
    }
    iteration = 0
    for iteration in tqdm(range(max_iter)):
        start_time = time.time()
        grad = newton_gradient(dual_vars, a, b, C, epsilon)
        grad_norm = np.linalg.norm(grad)
        hess = newton_hessian(dual_vars, a, b, C, epsilon)
        S = generate_S(2*n, sketch_dim)
        grad_S = S.T @ grad
        hess_S = S.T @ hess @ S
        def sub_p(z):
            return cubic_model(z, grad_S, hess_S, M)
        def sub_jac(z):
            return grad_cubic_model(z, grad_S, hess_S, M)
        newton_dir = S @ minimize(sub_p, np.zeros(sketch_dim), method='trust-constr',
                                  jac=sub_jac, options={'maxiter': 30}).x
        alpha = 1.0
        c = 0.5
        max_backtrack = 100
        current_obj = newton_objective(dual_vars, a, b, C, epsilon)
        for _ in range(max_backtrack):
            new_dual_vars = dual_vars + alpha * newton_dir
            new_obj = newton_objective(new_dual_vars, a, b, C, epsilon)
            if new_obj < current_obj:
                dual_vars = new_dual_vars
                break
            alpha *= c
        else:
            dual_vars += 1e-4 * newton_dir
        f = dual_vars[:n]
        g = dual_vars[n:]
        K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
        K_sum = K.sum()
        if K_sum > 0:
            P = K / K_sum
        else:
            P = np.zeros((n, n))
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        current_metrics = compute_metrics(P, a, b, C, epsilon)
        end_time = time.time()
        metrics['iterations'].append(iteration)
        if iteration % 10 == 0: print(grad_norm)
        metrics['error'].append(grad_norm)
        metrics['cost'].append(current_metrics['cost'])
        metrics['entropy'].append(current_metrics['entropy'])
        metrics['primal_obj'].append(current_metrics['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(current_metrics['marginal_a_violation'])
        metrics['marginal_b_violation'].append(current_metrics['marginal_b_violation'])
        metrics['total_marginal_violation'].append(current_metrics['total_marginal_violation'])
        metrics['gradient_norm'].append(grad_norm)
        metrics['time'].append((metrics['time'][-1]) + (end_time - start_time))
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    computation_time = 0
    for t in range(len(metrics['time'])):
        computation_time += metrics['time'][t]
    iterations = iteration + 1
    f = dual_vars[:n]
    g = dual_vars[n:]
    K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
    P = K / K.sum()
    return P, metrics, computation_time, iterations


def agd_method(a, b, C, epsilon, tol=1e-8, max_iter=300,
               warm_start_iter=0, alpha=0.001, momentum=True, max_time=None):
    n = len(a)
    dual_vars = np.zeros(2*n)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for _ in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f, g = epsilon*np.log(u), epsilon*np.log(v)
        dual_vars = np.concatenate([f, g])
    y = dual_vars.copy()
    t = 1.0
    metrics = {k: [] for k in [
        'iterations', 'error', 'cost', 'entropy', 'primal_obj', 'dual_obj',
        'marginal_a_violation', 'marginal_b_violation',
        'total_marginal_violation', 'gradient_norm', 'time'
    ]}
    metrics['time'].append(0.0)
    for it in tqdm(range(max_iter)):
        start = time.time()
        grad = newton_gradient(y, a, b, C, epsilon)
        gnorm = np.linalg.norm(grad)
        c = 0.5
        min_alpha = 1e-8
        current_obj = newton_objective(y, a, b, C, epsilon)
        for _ in range(50):
            candidate = y - alpha * grad
            cand_obj = newton_objective(candidate, a, b, C, epsilon)
            if cand_obj < current_obj:
                new_dual = candidate
                break
            alpha *= c
            if alpha < min_alpha:
                new_dual = y - min_alpha * grad
                break
        if momentum:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t*t))
            y = new_dual + ((t - 1)/t_new) * (new_dual - dual_vars)
            t = t_new
        else:
            y = new_dual
        dual_vars = new_dual
        f, g = dual_vars[:n], dual_vars[n:]
        K = np.exp((f[:, None] + g[None, :] - C) / epsilon)
        P = K / K.sum() if K.sum() > 0 else np.zeros_like(K)
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        cm = compute_metrics(P, a, b, C, epsilon)
        end = time.time()
        metrics['iterations'].append(it)
        if it % 100 == 0: print(gnorm)
        metrics['error'].append(gnorm)
        metrics['cost'].append(cm['cost'])
        metrics['entropy'].append(cm['entropy'])
        metrics['primal_obj'].append(cm['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(cm['marginal_a_violation'])
        metrics['marginal_b_violation'].append(cm['marginal_b_violation'])
        metrics['total_marginal_violation'].append(cm['total_marginal_violation'])
        metrics['gradient_norm'].append(gnorm)
        metrics['time'].append((metrics['time'][-1]) + (end - start))
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if gnorm < tol:
            break
    metrics['time'].pop()
    return P, metrics, metrics['time'][-1], it + 1


def lbfgs_method(a, b, C, epsilon, tol=1e-8, max_iter=300,
                 warm_start_iter=0, m=20, H0=0.001, max_time=None):
    n = len(a)
    dual_vars = np.zeros(2 * n)
    if warm_start_iter > 0:
        K = np.exp(-C / epsilon)
        u = np.ones(n)
        v = np.ones(n)
        for _ in range(warm_start_iter):
            v = b / (K.T @ u)
            u = a / (K @ v)
        f = epsilon * np.log(u)
        g = epsilon * np.log(v)
        dual_vars = np.concatenate([f, g])
    s_list, y_list = [], []
    metrics = {k: [] for k in [
        'iterations', 'error', 'cost', 'entropy', 'primal_obj', 'dual_obj',
        'marginal_a_violation', 'marginal_b_violation',
        'total_marginal_violation', 'gradient_norm', 'alpha', 'time'
    ]}
    metrics['time'].append(0.0)
    start_all = time.time()
    for iteration in tqdm(range(max_iter)):
        iter_start = time.time()
        grad = newton_gradient(dual_vars, a, b, C, epsilon)
        grad_norm = np.linalg.norm(grad)
        obj_curr = newton_objective(dual_vars, a, b, C, epsilon)
        q = grad.copy()
        k = len(s_list)
        if k > 0:
            alpha = np.zeros(k)
            rho = np.zeros(k)
            for i in range(k-1, -1, -1):
                rho[i] = 1.0 / (y_list[i] @ s_list[i])
                alpha[i] = rho[i] * (s_list[i] @ q)
                q = q - alpha[i] * y_list[i]
            r = H0 * q
            for i in range(k):
                beta = rho[i] * (y_list[i] @ r)
                r = r + s_list[i] * (alpha[i] - beta)
        else:
            r = H0 * q
        direction = -r
        alpha_step = 1.0
        c = 0.5
        max_backtrack = 50
        armijo_tau = 1e-4
        grad_dot_dir = grad @ direction
        for _ in range(max_backtrack):
            candidate = dual_vars + alpha_step * direction
            obj_new = newton_objective(candidate, a, b, C, epsilon)
            if obj_new <= obj_curr + armijo_tau * alpha_step * grad_dot_dir:
                break
            alpha_step *= c
        else:
            alpha_step = 1.0 * (c ** max_backtrack)
        new_vars = dual_vars + alpha_step * direction
        new_grad = newton_gradient(new_vars, a, b, C, epsilon)
        s_vec = new_vars - dual_vars
        y_vec = new_grad - grad
        if s_vec @ y_vec > 1e-12:
            s_list.append(s_vec)
            y_list.append(y_vec)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        dual_vars = new_vars
        f = dual_vars[:n]
        g = dual_vars[n:]
        Kmat = np.exp((f[:, None] + g[None, :] - C) / epsilon)
        Ksum = Kmat.sum()
        P = Kmat / Ksum if Ksum > 0 else np.zeros_like(Kmat)
        dual_obj = -newton_objective(dual_vars, a, b, C, epsilon)
        cm = compute_metrics(P, a, b, C, epsilon)
        iter_end = time.time()
        elapsed = iter_end - iter_start
        metrics['iterations'].append(iteration)
        if iteration % 100 == 0: print(grad_norm)
        metrics['error'].append(grad_norm)
        metrics['cost'].append(cm['cost'])
        metrics['entropy'].append(cm['entropy'])
        metrics['primal_obj'].append(cm['primal_obj'])
        metrics['dual_obj'].append(dual_obj)
        metrics['marginal_a_violation'].append(cm['marginal_a_violation'])
        metrics['marginal_b_violation'].append(cm['marginal_b_violation'])
        metrics['total_marginal_violation'].append(cm['total_marginal_violation'])
        metrics['gradient_norm'].append(grad_norm)
        metrics['alpha'].append(alpha_step)
        metrics['time'].append((metrics['time'][-1]) + elapsed)
        if max_time is not None and metrics['time'][-1] >= max_time:
            break
        if grad_norm < tol:
            break
    metrics['time'].pop()
    total_time = time.time() - start_all
    return P, metrics, total_time, iteration + 1
