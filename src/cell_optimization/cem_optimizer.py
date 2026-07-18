import numpy as np

class CrossEntropyOptimizer:
    def __init__(
        self,
        population_size=64,
        elite_fraction=0.15,
        iterations=15,
        smoothing=0.7,
        min_std=1e-4
    ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.iterations = iterations
        self.smoothing = smoothing
        self.min_std = min_std

    def optimize(self, evaluator_func, x0, bounds, active_indices, G_vector, verbose=True):
        """
        Sensitivity-Guided Cross-Entropy Method (SG-CEM) Optimizer.
        - evaluator_func: takes full x vector, returns float score (smaller is better) and feasibility boolean.
        - x0: baseline full design vector.
        - bounds: DESIGN_BOUNDS (shape: (8, 2)).
        - active_indices: list of indices of parameters currently being optimized.
        - G_vector: Jacobian or sensitivity vector for ALL parameters (shape: (8,)).
        """
        xl_full, xu_full = bounds[:, 0], bounds[:, 1]
        xl = xl_full[active_indices]
        xu = xu_full[active_indices]
        R = xu - xl # bounding ranges for active variables

        # 1. Sensitivity-Weighted Initialization
        G_active = np.abs(G_vector[active_indices])
        max_g = np.max(G_active) if np.max(G_active) > 0 else 1.0
        w = G_active / max_g

        # Highly sensitive variables -> smaller variance; insensitive -> larger variance
        sigma_max = 0.25
        sigma_min = 0.02
        std_fractions = (1.0 - w) * sigma_max + w * sigma_min

        # Initial mean and covariance
        mu = x0[active_indices].copy()
        cov = np.diag((std_fractions * R) ** 2)
        initial_std = np.sqrt(np.diag(cov))

        best_score = 1e12
        best_x = mu.copy()

        # Track history of best objective for convergence
        best_history = []

        for it in range(self.iterations):
            # 2. Adaptive Population Size
            max_std_ratio = np.max(np.sqrt(np.diag(cov)) / (initial_std + 1e-12))
            if max_std_ratio > 0.5:
                pop_size = self.population_size
            elif max_std_ratio > 0.2:
                pop_size = max(32, int(self.population_size / 2))
            elif max_std_ratio > 0.1:
                pop_size = max(16, int(self.population_size / 4))
            else:
                pop_size = max(8, int(self.population_size / 8))

            # 3. Covariance Regularization
            cov_reg = cov + np.diag((self.min_std * R) ** 2)

            # 4. Sample candidates
            try:
                samples = np.random.multivariate_normal(mu, cov_reg, size=pop_size)
            except np.linalg.LinAlgError:
                stds = np.sqrt(np.maximum(np.diag(cov_reg), 1e-12))
                samples = np.random.normal(mu, stds, size=(pop_size, len(active_indices)))

            # Clamp to bounds
            samples = np.clip(samples, xl, xu)

            # 5. Evaluate candidates
            scores = []
            valid_samples = []

            for sample in samples:
                # Reconstruct full design vector
                x_full = x0.copy()
                x_full[active_indices] = sample

                # Geometry-aware rounding
                for idx, val in enumerate(x_full):
                    if idx in [0, 1]:
                        x_full[idx] = np.round(val * 1e6) / 1e6
                    elif idx in [4, 5]:
                        x_full[idx] = np.round(val * 1e8) / 1e8

                score, feasible = evaluator_func(x_full)
                if not feasible:
                    score = 1e12
                scores.append(score)
                valid_samples.append(sample)

            scores = np.array(scores)
            valid_samples = np.array(valid_samples)

            # Sort samples by score
            indices = np.argsort(scores)
            sorted_scores = scores[indices]
            sorted_samples = valid_samples[indices]

            # 6. Select elite samples
            elite_count = max(2, int(pop_size * self.elite_fraction))
            feasible_mask = sorted_scores < 1e10
            feasible_count = np.sum(feasible_mask)

            if feasible_count >= 2:
                actual_elite_count = min(elite_count, feasible_count)
                elites = sorted_samples[:actual_elite_count]
                elite_scores = sorted_scores[:actual_elite_count]
            else:
                elites = sorted_samples[:elite_count]
                elite_scores = sorted_scores[:elite_count]

            if elite_scores[0] < best_score:
                best_score = elite_scores[0]
                best_x = elites[0].copy()

            # 7. Elite Diversity Check
            if len(elites) >= 2:
                elite_std = np.std(elites, axis=0)
                if np.max(elite_std / R) < 0.005:
                    if verbose:
                        print(f"INFO[CEM]: Elite diversity collapse detected. Boosting covariance.")
                    cov += np.diag((0.1 * initial_std) ** 2)

            # 8. Update distribution parameters (mean and covariance)
            if len(elites) >= 2:
                new_mu = np.mean(elites, axis=0)
                diff = elites - new_mu
                new_cov = (diff.T @ diff) / len(elites)

                mu = self.smoothing * new_mu + (1.0 - self.smoothing) * mu
                cov = self.smoothing * new_cov + (1.0 - self.smoothing) * cov
            else:
                mu = 0.5 * sorted_samples[0] + 0.5 * mu

            if verbose:
                print(f"INFO[CEM]: Iteration {it+1}/{self.iterations} - Best Score: {best_score:.6f} - Feasible count: {feasible_count}/{pop_size}")

            # Convergence check
            best_history.append(best_score)
            max_std = np.max(np.sqrt(np.diag(cov)) / R)
            if max_std < self.min_std:
                if verbose:
                    print(f"INFO[CEM]: Converged on max std of covariance: {max_std:.6e} < {self.min_std}")
                break

            if len(best_history) >= 5 and np.abs(best_history[-1] - best_history[-5]) < 1e-5:
                if verbose:
                    print(f"INFO[CEM]: Converged on stable best objective score: {best_score:.6f}")
                break

        return best_x
