import numpy as np

# --- 1. Material Properties (Core References) ---
F = 96485.332

# NFPP Cathode: Na2FeP2O7
# Ref: paper.md, ResearchGate (10.1021/acssuschemeng.7b04516)
NFPP_MOLAR_MASS = 275.77e-3 # [kg/mol]
NFPP_DENSITY = 3200.0        # [kg/m3]
NFPP_SPECIFIC_CAPACITY_MAH_G = 97.19

# Hard Carbon Anode
# Ref: MTI, Kuraray, Ossila
HC_DENSITY = 1500.0          # [kg/m3]
HC_PRACTICAL_CAPACITY_MAH_G = 300.0

# Additive Densities
# Ref: Typical values
CARBON_DENSITY = 2000.0      # [kg/m3]
BINDER_DENSITY = 1780.0      # [kg/m3] (PVDF)

def compute_volume_fractions(wt_am, wt_c, wt_b, rho_am, rho_c, rho_b, porosity):
    v_am = wt_am / rho_am
    v_c = wt_c / rho_c
    v_b = wt_b / rho_b
    v_total_solid = v_am + v_c + v_b
    eps_am = (1 - porosity) * (v_am / v_total_solid)
    return eps_am

def compute_parameters():
    c_max_p = NFPP_DENSITY / NFPP_MOLAR_MASS
    # Corrected: 300 Ah/kg * 1500 kg/m3 * 3600 s/h / 96485 C/mol
    c_max_hc = (HC_PRACTICAL_CAPACITY_MAH_G * HC_DENSITY * 3600.0) / F

    eps_am_p = compute_volume_fractions(0.85, 0.08, 0.07, NFPP_DENSITY, CARBON_DENSITY, BINDER_DENSITY, 0.3)
    eps_am_n = compute_volume_fractions(0.88, 0.06, 0.06, HC_DENSITY, CARBON_DENSITY, BINDER_DENSITY, 0.3)

    area = 0.130 * 0.070
    L_p = 0.0001
    cap_layer = (area * L_p * eps_am_p * c_max_p * F) / 3600
    n_layers = 10.0 / cap_layer

    return {
        "Positive max concentration [mol.m-3]": c_max_p,
        "Negative max concentration [mol.m-3]": c_max_hc,
        "Positive AM volume fraction": eps_am_p,
        "Negative AM volume fraction": eps_am_n,
        "Capacity per layer [A.h]": cap_layer,
        "Recommended N_layers": int(np.ceil(n_layers)),
    }

if __name__ == "__main__":
    params = compute_parameters()
    for k, v in params.items():
        print(f"{k}: {v:.4f}")
