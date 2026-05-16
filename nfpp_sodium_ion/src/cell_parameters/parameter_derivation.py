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

def compute_parameters():
    # c_max [mol/m3]
    # NFPP: c_max = rho / M
    c_max_p = NFPP_DENSITY / NFPP_MOLAR_MASS

    # Hard Carbon: c_max = (Cap_mAh_g * rho * 3600) / F
    # Note: mAh/g is equivalent to Ah/kg.
    # c_max = (300 Ah/kg * 1500 kg/m3 * 3600 s/h) / 96485 C/mol = 16790.1 mol/m3
    c_max_hc = (HC_PRACTICAL_CAPACITY_MAH_G * HC_DENSITY * 3600.0) / F

    return {
        "Positive max concentration [mol.m-3]": c_max_p,
        "Negative max concentration [mol.m-3]": c_max_hc,
    }

if __name__ == "__main__":
    params = compute_parameters()
    for k, v in params.items():
        print(f"{k}: {v:.4f}")
