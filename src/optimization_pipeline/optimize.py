import numpy as np
from scipy.optimize import minimize
import pybamm
from nfpp_sodium_ion.src.cell_parameters.cell_alpha import get_parameter_values

class NFPPoptimizer:
    """
    NFPP Cell Optimization Pipeline (Physics-Consistent).
    1. Electrolyte Optimization (Cost)
    2. Numerical Sensitivity-Driven Design Space Reduction
    3. Physics-Coupled Multi-Objective Optimization (SLSQP)
    """

    def __init__(self):
        self.base_params = get_parameter_values()
        self.theta_names = [
            "Positive electrode thickness [m]",
            "Negative electrode thickness [m]",
            "Positive electrode porosity",
            "Negative electrode porosity",
            "Positive particle radius [m]"
        ]
        self.theta_initial = np.array([0.0001, 0.00012, 0.3, 0.3, 1e-6])
        self.bounds = [(5e-5, 2e-4), (5e-5, 2e-4), (0.1, 0.4), (0.1, 0.4), (1e-7, 1e-5)]

    def step1_electrolyte_optimization(self):
        """
        Step 1: Minimizes electrolyte cost based on salt and additives.
        """
        print("Stage 1: Running Electrolyte Cost Optimization...")
        def cost_fn(x):
            napf6, nadfob, fec, vc = x
            return 10*napf6 + 25*nadfob + 15*fec + 20*vc

        bounds = [(0.8, 1.2), (0.1, 0.5), (1.0, 5.0), (1.0, 4.0)]
        res = minimize(cost_fn, [1.0, 0.2, 3.0, 2.0], bounds=bounds)
        print(f"  Optimal Electrolyte: {res.x}")
        return res.x

    def physics_objective(self, theta, active_indices=None, reduced_val=None):
        """
        Multi-objective cost J = -alpha*E + beta*R + gamma*T + delta*D
        """
        full_theta = np.array(theta, copy=True)
        if active_indices is not None and reduced_val is not None:
            full_theta[active_indices] = reduced_val

        L_c, L_a, eps_c, eps_a, r_p = full_theta
        # Physics surrogates (Energy, Resistance, Peak T, Degradation)
        capacity = (L_c * (1-eps_c)) * 600
        resistance = (L_c / 50.0) + (r_p**2 / 1e-12)
        peak_t = 100 * (resistance + 0.01)
        degradation = 0.01 * np.exp(0.1 * (peak_t - 25))

        alpha, beta, gamma, delta = 1.0, 0.8, 0.5, 0.3
        return -(alpha * capacity) + (beta * resistance) + (gamma * peak_t) + (delta * degradation)

    def run_sensitivity_analysis(self):
        """
        Calculates numerical sensitivities dJ/dtheta to identify influential parameters.
        """
        print("Stage 2.1: Running Numerical Sensitivity Analysis...")
        epsilon = 1e-6
        grad = np.zeros_like(self.theta_initial)
        j_base = self.physics_objective(self.theta_initial)

        for i in range(len(self.theta_initial)):
            theta_plus = np.array(self.theta_initial, copy=True)
            theta_plus[i] += epsilon
            grad[i] = (self.physics_objective(theta_plus) - j_base) / epsilon

        sensitivities = np.abs(grad * self.theta_initial) # Elasticity
        active_indices = np.where(sensitivities > 0.05 * np.max(sensitivities))[0]
        print(f"  Influential parameters identified: {[self.theta_names[i] for i in active_indices]}")
        return active_indices

    def run_optimization(self):
        print("Stage 2: Running Physics-Consistent Optimization...")
        opt_electrolyte = self.step1_electrolyte_optimization()
        active_idx = self.run_sensitivity_analysis()

        def np_ratio_con(x_reduced):
            theta = np.array(self.theta_initial, copy=True)
            theta[active_idx] = x_reduced
            return theta[1]*(1-theta[3])/(theta[0]*(1-theta[2])) - 1.05

        cons = ({'type': 'ineq', 'fun': np_ratio_con})
        res = minimize(
            lambda x: self.physics_objective(self.theta_initial, active_idx, x),
            self.theta_initial[active_idx],
            method='SLSQP',
            bounds=[self.bounds[i] for i in active_idx],
            constraints=cons
        )

        optimal_theta = np.array(self.theta_initial, copy=True)
        optimal_theta[active_idx] = res.x
        print(f"  Optimal Design Solution: {optimal_theta}")
        return {"electrolyte": opt_electrolyte, "design": optimal_theta}

if __name__ == "__main__":
    optimizer = NFPPoptimizer()
    optimizer.run_optimization()
