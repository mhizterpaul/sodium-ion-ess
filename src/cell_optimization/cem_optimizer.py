import numpy as np
from concurrent.futures import ThreadPoolExecutor

class CrossEntropyOptimizer:
    def __init__(
        self,
        population_size=64,
        elite_fraction=0.15,
        iterations=15,
        smoothing=0.7,
        min_std=1e-4,
        lambda_penalty=1e5
    ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.iterations = iterations
        self.smoothing = smoothing
        self.min_std = min_std
        self.lambda_penalty = lambda_penalty
        self.cache = {}

    def _to_z(self, x, xl, xu):
        range_val = np.maximum(xu - xl, 1e-12)
        return (x - xl) / range_val

    def _to_x(self, z, xl, xu):
        return xl + z * (xu - xl)

    def _truncated_sample(self, mu, cov, size):
        """
        Draw samples in z-space [0, 1]^d.
        If a sample violates bounds, repeatedly resample the violating dimensions
        up to 10 times. Fallback to clipping if still out of bounds.
        """
        d = len(mu)
        samples = np.zeros((size, d))

        for i in range(size):
            success = False
            for attempt in range(10):
                try:
                    s = np.random.multivariate_normal(mu, cov)
                except np.linalg.LinAlgError:
                    stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
                    s = np.random.normal(mu, stds)

                if np.all(s >= 0.0) and np.all(s <= 1.0):
                    samples[i] = s
                    success = True
                    break

            if not success:
                samples[i] = np.clip(s, 0.0, 1.0)

        return samples

    def optimize(self, evaluator_func, x0, bounds, active_indices, G_vector, verbose=True):
        """
        Sensitivity-Guided Cross-Entropy Method (SG-CEM) Optimizer.
        """
        xl_full, xu_full = bounds[:, 0], bounds[:, 1]
        xl = xl_full[active_indices]
        xu = xu_full[active_indices]

        # 1. Sensitivity-Weighted Initialization
        G_active = np.abs(G_vector[active_indices])
        max_g = np.max(G_active) if np.max(G_active) > 0 else 1.0
        w_sens = G_active / max_g

        sigma_max = 0.25
        sigma_min = 0.02
        std_fractions = (1.0 - w_sens) * sigma_max + w_sens * sigma_min

        mu_z = self._to_z(x0[active_indices], xl, xu)
        cov_z = np.diag(std_fractions ** 2)
        initial_std_z = np.sqrt(np.diag(cov_z))

        best_score = 1e12
        best_x = x0[active_indices].copy()
        best_history = []

        for it in range(self.iterations):
            # 2. Adaptive Population Size
            max_std_ratio = np.max(np.sqrt(np.diag(cov_z)) / (initial_std_z + 1e-12))
            if max_std_ratio > 0.5:
                pop_size = self.population_size
            elif max_std_ratio > 0.2:
                pop_size = max(32, int(self.population_size / 2))
            elif max_std_ratio > 0.1:
                pop_size = max(16, int(self.population_size / 4))
            else:
                pop_size = max(8, int(self.population_size / 8))

            # 3. Covariance Regularization / Positive Definiteness Check via Eigenvalue Decomposition
            eigvals, eigvecs = np.linalg.eigh(cov_z)
            eigvals = np.maximum(eigvals, self.min_std**2)
            cov_z_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # 4. Truncated sampling in z-space
            samples_z = self._truncated_sample(mu_z, cov_z_reg, pop_size)

            # 5. Candidate Evaluation with caching & geometry-aware rounding
            def evaluate_one(sample_z):
                cache_key = tuple(np.round(sample_z, 4))
                if cache_key in self.cache:
                    return self.cache[cache_key]

                x_active = self._to_x(sample_z, xl, xu)
                x_full = x0.copy()
                x_full[active_indices] = x_active

                # Geometry-aware rounding before evaluation
                for idx, val in enumerate(x_full):
                    if idx in [0, 1]:
                        x_full[idx] = np.round(val * 1e6) / 1e6
                    elif idx in [4, 5]:
                        x_full[idx] = np.round(val * 1e8) / 1e8

                res_eval = evaluator_func(x_full)
                if isinstance(res_eval, tuple):
                    if len(res_eval) == 3:
                        obj_val, g_list, feasible = res_eval
                    else:
                        obj_val, feasible = res_eval
                        g_list = [0.0] if feasible else [1.0]
                else:
                    obj_val = res_eval
                    feasible = True
                    g_list = [0.0]

                # Penalize infeasible samples: f_penalized = f + lambda * sum(max(0, g)^2)
                penalty = 0.0
                if g_list:
                    penalty = self.lambda_penalty * sum(max(0.0, g)**2 for g in g_list)
                    if not feasible and penalty == 0.0:
                        penalty = 1e5

                score = obj_val + penalty
                result = (score, feasible)
                self.cache[cache_key] = result
                return result

            # Run parallel evaluations
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(evaluate_one, samples_z))

            scores = np.array([r[0] for r in results])
            feasibles = np.array([r[1] for r in results])

            indices = np.argsort(scores)
            sorted_scores = scores[indices]
            sorted_samples_z = samples_z[indices]
            sorted_feasibles = feasibles[indices]

            # 6. Adaptive Elite Fraction
            progress = it / self.iterations
            if progress < 0.3:
                elite_frac = 0.25
            elif progress < 0.7:
                elite_frac = 0.15
            else:
                elite_frac = 0.05

            elite_count = max(2, int(pop_size * elite_frac))
            elites_z = sorted_samples_z[:elite_count]
            elite_scores = sorted_scores[:elite_count]

            if elite_scores[0] < best_score:
                best_score = elite_scores[0]
                best_x = self._to_x(elites_z[0], xl, xu)

            # 7. Elite Diversity Check
            if len(elites_z) >= 2:
                elite_std = np.std(elites_z, axis=0)
                if np.max(elite_std) < 0.005:
                    if verbose:
                        print(f"INFO[CEM]: Elite diversity collapse detected. Boosting covariance.")
                    cov_z += np.diag((0.1 * initial_std_z) ** 2)

            # 8. Update distribution parameters (Weighted Elites & Jacobian contraction)
            if len(elites_z) >= 2:
                min_es = np.min(elite_scores)
                max_es = np.max(elite_scores)
                range_es = max_es - min_es
                if range_es > 1e-12:
                    norm_scores = (elite_scores - min_es) / range_es
                else:
                    norm_scores = np.zeros_like(elite_scores)

                w = np.exp(-norm_scores)
                w /= np.sum(w)

                new_mu_z = np.sum(w[:, None] * elites_z, axis=0)

                diff = elites_z - new_mu_z
                new_cov_z = np.zeros_like(cov_z)
                for j in range(len(elites_z)):
                    new_cov_z += w[j] * np.outer(diff[j], diff[j])

                mu_z = self.smoothing * new_mu_z + (1.0 - self.smoothing) * mu_z
                cov_z = self.smoothing * new_cov_z + (1.0 - self.smoothing) * cov_z

                # Jacobian incorporation: shrink covariance faster in highly sensitive dimensions
                alpha_jac = 0.1
                cov_z = cov_z / (1.0 + alpha_jac * w_sens[:, None] * w_sens[None, :])
            else:
                mu_z = 0.5 * sorted_samples_z[0] + 0.5 * mu_z

            if verbose:
                num_feas = np.sum(feasibles)
                print(f"INFO[CEM]: Iteration {it+1}/{self.iterations} - Best Score: {best_score:.6f} - Feasible: {num_feas}/{pop_size}")

            best_history.append(best_score)
            max_std = np.max(np.sqrt(np.diag(cov_z)))
            if max_std < self.min_std:
                if verbose:
                    print(f"INFO[CEM]: Converged on max std of covariance: {max_std:.6e} < {self.min_std}")
                break

            if len(best_history) >= 5 and np.abs(best_history[-1] - best_history[-5]) < 1e-5:
                if verbose:
                    print(f"INFO[CEM]: Converged on stable best objective score: {best_score:.6f}")
                break

        return best_x
