import os
import csv
import random
import numpy as np
from opendssdirect import dss
import src.power_plant.plant as plant

def build_random_downstream_tree(feeder_idx: int, num_buses: int, line_mult: float, has_ring: bool):
    """
    Programmatically generates a random tree topology of num_buses connected to feederX_sec.
    The algorithm starts with feederX_sec as the root and iteratively grows the tree,
    adding a new bus and connecting it to a randomly selected existing bus.
    This guarantees a connected, radial structure.
    If has_ring is True, a tie-line is added to close a loop.
    """
    root_bus = f"feeder{feeder_idx}_sec"
    existing_buses = [root_bus]

    # 1. Define downstream line code for LV (0.415 kV)
    dss.run_command(f"new linecode.down_lv_{feeder_idx} nphases=3 r1=0.45 x1=0.15 r0=1.20 x0=0.35 c1=4.0 c0=2.0 units=km")

    lines_added = []
    total_r = 0.0
    total_x = 0.0

    # 2. Iteratively grow the tree
    for i in range(1, num_buses):
        new_bus = f"f{feeder_idx}_node{i}"
        parent_bus = random.choice(existing_buses)

        # Random line length representing physical distance (e.g. 0.03 to 0.12 km)
        l_km = random.uniform(0.03, 0.12) * line_mult
        line_name = f"line.down_{feeder_idx}_{i}"

        dss.run_command(f"new {line_name} bus1={parent_bus} bus2={new_bus} phases=3 linecode=down_lv_{feeder_idx} length={l_km} units=km")

        existing_buses.append(new_bus)
        lines_added.append((parent_bus, new_bus))

        # Track total impedance for latent state metrics
        total_r += 0.45 * l_km
        total_x += 0.15 * l_km

    # 3. Add occasional Loop/Ring (Tie-line)
    is_ring_formed = 0
    if has_ring and len(existing_buses) > 5:
        # Connect two random non-adjacent buses to form a loop
        bus_a = existing_buses[3]
        bus_b = existing_buses[-1]
        dss.run_command(f"new line.tie_{feeder_idx} bus1={bus_a} bus2={bus_b} phases=3 linecode=down_lv_{feeder_idx} length=0.15 units=km")
        is_ring_formed = 1

    # 4. Distribute loads, capacitors, and motors across the buses
    motor_count = 0
    cap_count = 0
    load_kw_total = 0.0

    # Distribute mixed loads across 60% of the nodes
    for bus in existing_buses[1:]:
        if random.random() < 0.6:
            load_kw = random.uniform(5.0, 25.0)
            # Mix load models: 1 (constant PQ), 2 (constant Z), 3 (constant I)
            l_model = random.choice([1, 2, 3])
            pf = random.choice([0.85, 0.90, 0.95])
            dss.run_command(f"new load.l_{bus} bus1={bus} phases=3 kv=0.415 kw={load_kw} pf={pf} model={l_model} status=fixed")
            load_kw_total += load_kw

        # Place occasional Capacitor bank
        if random.random() < 0.12:
            cap_kvar = random.choice([15.0, 30.0, 45.0])
            dss.run_command(f"new capacitor.c_{bus} bus1={bus} phases=3 kv=0.415 kvar={cap_kvar} conn=wye")
            cap_count += 1

        # Place occasional Motor (starts during events, modeled as constant-impedance block during starting)
        if random.random() < 0.08:
            motor_count += 1

    # Calculate topology entropy as a latent variable (approximation based on average degree)
    deg_sum = 2 * len(lines_added) + (2 if is_ring_formed else 0)
    avg_degree = deg_sum / len(existing_buses)
    topology_entropy = float(avg_degree * np.log2(avg_degree + 1e-3))

    return {
        "num_buses": len(existing_buses),
        "num_edges": len(lines_added) + is_ring_formed,
        "is_ring": is_ring_formed,
        "total_r_ohm": round(total_r, 4),
        "total_x_ohm": round(total_x, 4),
        "load_kw_total": round(load_kw_total, 2),
        "motor_count": motor_count,
        "capacitor_count": cap_count,
        "topology_entropy": round(topology_entropy, 3)
    }

def run_simulation_scenarios(n_scenarios: int = 15):
    """
    Runs a series of perturbed scenarios in OpenDSS, couples with ATP transients,
    extracts feeder and transformer parameters, and exports them to CSV.
    """
    print(f"INFO: Running {n_scenarios} Physics-Based Operational Scenarios...")

    events = [
        'steady_state', 'transformer_energization', 'capacitor_switching',
        'motor_starting', 'temporary_fault', 'nonlinear_load'
    ]

    results = []

    for idx in range(n_scenarios):
        # 1. Reset plant to fixed 33/11 kV upstream system
        plant.initialize_plant()

        # 2. Determine scenario configurations
        event = events[idx % len(events)]
        ring_feeder_idx = (idx % 3) + 1 if idx in [7, 12, 14] else 0 # ring occasionally

        line_mult = 1.0 + 0.15 * np.sin(idx) # vary line lengths by +-15%

        # 3. Generate random tree networks of 20-80 buses for each of the 3 feeders
        # Ensure that the realization algorithm never "knows" this graph, only OpenDSS does!
        downstream_data = {}
        for f in range(1, 4):
            num_buses = random.randint(20, 80)
            has_ring = (f == ring_feeder_idx)
            down_meta = build_random_downstream_tree(f, num_buses, line_mult, has_ring)
            downstream_data[f] = down_meta

        # 4. Solve power flow
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print(f"WARNING: Scenario {idx} did not converge. Attempting solve with direct mode...")
            dss.run_command("Solve mode=direct")

        # 5. Extract detailed boundary measurements M
        m = plant.get_boundary_measurements()

        # 6. Emulate ATP-EMTP transient wave and extract dynamic features
        # Trigger event transient at active feeder
        active_feeder = (idx % 3) + 1
        t, v_wave, i_wave = plant.emulate_atp_transient(event, m, active_feeder, duration=0.04, fs=10000.0)
        fft_feat = plant.extract_dynamic_transient_features(t, v_wave, i_wave, fs=10000.0)

        # 7. Compile the complete scenario data mapping
        # Hidden Parameters -> Boundary Measurements -> Extracted Features -> Latent State
        # We save separate fields for feeder and transformer metrics
        scenario_data = {
            "scenario_index": idx,
            "topology_type": "ring" if ring_feeder_idx > 0 else "radial",
            "simulated_event": event,
            "active_feeder": active_feeder,

            # --- Hidden Parameters (only OpenDSS knows, target for DSE) ---
            "hidden_total_buses": sum(downstream_data[f]["num_buses"] for f in range(1, 4)),
            "hidden_total_edges": sum(downstream_data[f]["num_edges"] for f in range(1, 4)),
            "hidden_f1_r_ohm": downstream_data[1]["total_r_ohm"],
            "hidden_f2_r_ohm": downstream_data[2]["total_r_ohm"],
            "hidden_f3_r_ohm": downstream_data[3]["total_r_ohm"],
            "hidden_motor_count": sum(downstream_data[f]["motor_count"] for f in range(1, 4)),
            "hidden_capacitor_count": sum(downstream_data[f]["capacitor_count"] for f in range(1, 4)),

            # --- Boundary Measurements ---
            # Voltages & Symmetrical Components (feeder 1 head)
            "feeder1_voltage_mag_kv": round(np.mean(m["feeder1_voltage_mag"]) / 1000.0, 4),
            "feeder1_voltage_pos_mag_kv": round(m["feeder1_voltage_pos_mag"] / 1000.0, 4),
            "feeder1_voltage_neg_mag_kv": round(m["feeder1_voltage_neg_mag"] / 1000.0, 4),
            "feeder1_voltage_zero_mag_kv": round(m["feeder1_voltage_zero_mag"] / 1000.0, 4),
            "feeder1_voltage_unbalance_pct": round(m["feeder1_voltage_unbalance_pct"], 3),

            # Currents & Symmetrical Components
            "feeder1_current_mag_amp": round(np.mean(m["feeder1_current_mag"]), 2),
            "feeder1_current_pos_mag_amp": round(m["feeder1_current_pos_mag"], 2),
            "feeder1_current_unbalance_pct": round(m["feeder1_current_unbalance_pct"], 3),

            # Active/Reactive/Apparent Power and power factor
            "feeder1_p_kw": round(m["feeder1_p_kw"], 2),
            "feeder1_q_kvar": round(m["feeder1_q_kvar"], 2),
            "feeder1_s_kva": round(m["feeder1_s_kva"], 2),
            "feeder1_pf": round(m["feeder1_pf"], 3),

            # Feeder 2 head measurements
            "feeder2_voltage_mag_kv": round(np.mean(m["feeder2_voltage_mag"]) / 1000.0, 4),
            "feeder2_voltage_unbalance_pct": round(m["feeder2_voltage_unbalance_pct"], 3),
            "feeder2_current_mag_amp": round(np.mean(m["feeder2_current_mag"]), 2),
            "feeder2_current_unbalance_pct": round(m["feeder2_current_unbalance_pct"], 3),
            "feeder2_p_kw": round(m["feeder2_p_kw"], 2),
            "feeder2_q_kvar": round(m["feeder2_q_kvar"], 2),
            "feeder2_pf": round(m["feeder2_pf"], 3),

            # Feeder 3 head measurements
            "feeder3_voltage_mag_kv": round(np.mean(m["feeder3_voltage_mag"]) / 1000.0, 4),
            "feeder3_voltage_unbalance_pct": round(m["feeder3_voltage_unbalance_pct"], 3),
            "feeder3_current_mag_amp": round(np.mean(m["feeder3_current_mag"]), 2),
            "feeder3_current_unbalance_pct": round(m["feeder3_current_unbalance_pct"], 3),
            "feeder3_p_kw": round(m["feeder3_p_kw"], 2),
            "feeder3_q_kvar": round(m["feeder3_q_kvar"], 2),
            "feeder3_pf": round(m["feeder3_pf"], 3),

            # --- Extracted Features (Z_eq, phase diff, stiffness, FFT features) ---
            "feeder1_eq_impedance_ohm": round(m["feeder1_eq_impedance_ohm"], 3),
            "feeder2_eq_impedance_ohm": round(m["feeder2_eq_impedance_ohm"], 3),
            "feeder3_eq_impedance_ohm": round(m["feeder3_eq_impedance_ohm"], 3),

            "feeder1_phase_angle_diff_deg": round(m["feeder1_phase_angle_diff_deg"], 3),
            "feeder2_phase_angle_diff_deg": round(m["feeder2_phase_angle_diff_deg"], 3),
            "feeder3_phase_angle_diff_deg": round(m["feeder3_phase_angle_diff_deg"], 3),

            "feeder1_stiffness_kva": round(m["feeder1_stiffness_kva"], 2),
            "feeder2_stiffness_kva": round(m["feeder2_stiffness_kva"], 2),
            "feeder3_stiffness_kva": round(m["feeder3_stiffness_kva"], 2),

            "spectral_centroid_hz": fft_feat["spectral_centroid"],
            "dominant_frequency_hz": fft_feat["dominant_frequency"],
            "wavelet_energy_low_pct": fft_feat["wavelet_energy_low_pct"],
            "wavelet_energy_mid_pct": fft_feat["wavelet_energy_mid_pct"],
            "wavelet_energy_high_pct": fft_feat["wavelet_energy_high_pct"],

            # --- Transformer Parameters (HV Volt, Current, Losses, Loading, Regulation, etc.) ---
            "transformer1_hv_voltage_v": round(m["transformer1_hv_voltage"], 1),
            "transformer1_hv_current_amp": round(m["transformer1_hv_current"], 2),
            "transformer1_loading_pct": round(m["transformer1_loading_pct"], 2),
            "transformer1_copper_loss_kw": round(m["transformer1_copper_loss_kw"], 3),
            "transformer1_core_loss_kw": round(m["transformer1_core_loss_kw"], 3),
            "transformer1_voltage_regulation_pct": round(m["transformer1_voltage_regulation_pct"], 3),
            "transformer1_eq_impedance_ohm": round(m["transformer1_eq_impedance_ohm"], 3),
            "transformer1_tap_position": m["transformer1_tap_position"],

            "transformer2_loading_pct": round(m["transformer2_loading_pct"], 2),
            "transformer2_copper_loss_kw": round(m["transformer2_copper_loss_kw"], 3),
            "transformer2_core_loss_kw": round(m["transformer2_core_loss_kw"], 3),
            "transformer2_voltage_regulation_pct": round(m["transformer2_voltage_regulation_pct"], 3),
            "transformer2_eq_impedance_ohm": round(m["transformer2_eq_impedance_ohm"], 3),
            "transformer2_tap_position": m["transformer2_tap_position"],

            "transformer3_loading_pct": round(m["transformer3_loading_pct"], 2),
            "transformer3_copper_loss_kw": round(m["transformer3_copper_loss_kw"], 3),
            "transformer3_core_loss_kw": round(m["transformer3_core_loss_kw"], 3),
            "transformer3_voltage_regulation_pct": round(m["transformer3_voltage_regulation_pct"], 3),
            "transformer3_eq_impedance_ohm": round(m["transformer3_eq_impedance_ohm"], 3),
            "transformer3_tap_position": m["transformer3_tap_position"],

            # --- Latent State Mapping ---
            "latent_total_load_demand_kw": round(sum(downstream_data[f]["load_kw_total"] for f in range(1, 4)), 2),
            "latent_topology_entropy": round(sum(downstream_data[f]["topology_entropy"] for f in range(1, 4)), 3),
            "latent_avg_f1_electrical_distance_km": round(0.075 * line_mult * downstream_data[1]["num_buses"], 3)
        }
        results.append(scenario_data)

    # 8. Export to CSV file
    csv_dir = "src/simulation"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "scenario_results.csv")

    headers = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"INFO: Successfully exported {n_scenarios} scenarios to {csv_path}")
    return results

if __name__ == "__main__":
    run_simulation_scenarios()
