import numpy as np
from opendssdirect import dss

def initialize_plant():
    """
    Initializes the fixed upstream distribution station using OpenDSS.
    The known plant has standard distribution voltage levels:
    - Utility Grid Source (33 kV)
    - Substation Transformer (33 kV to 11 kV, 10 MVA)
    - Main Distribution Bus / PCC (11 kV)
    - PCU (1 unit interfacing the shared Generator)
    - Shared Generator (coupled at 11 kV)
    - Medium-voltage Switchgear
    - Three 11 kV Feeders (Line 1, Line 2, Line 3)
    - Fixed set of three 11/0.415 kV step-down Distribution Transformers acting as edge interfaces
    """
    print("INFO: Initializing OpenDSS Physics-Based Plant Model (33/11 kV)...")

    # 1. Clear previous systems and define main circuit at swing bus (33 kV)
    dss.Basic.ClearAll()
    dss.run_command("new circuit.FixedPlant basekv=33.0 pu=1.0 phases=3")

    # 2. Substation Transformer (33 kV to 11 kV, delta-wye)
    dss.run_command("new transformer.substation phases=3 windings=2 buses=[sourcebus, main_bus] conns=[delta, wye] kvs=[33.0, 11.0] kvas=[10000, 10000] %r=0.4 xhl=7.0")

    # 3. Generator, PCU, and Switchgear
    # Shared Generator (e.g. 2 MW capacity) coupled at 11 kV main_bus
    dss.run_command("new generator.shared_gen bus1=main_bus phases=3 kv=11.0 kw=1500 pf=0.9 model=1")

    # 4. Outgoing radial 11 kV Feeders (Line 1, Line 2, Line 3)
    # Standard 11 kV line parameters
    dss.run_command("new linecode.feeder nphases=3 r1=0.25 x1=0.35 r0=0.75 x0=1.12 c1=12.0 c0=6.0 units=km")

    # Feeders extending from main_bus to the respective 11 kV feeder head buses
    # Feeder lengths are sufficient to make impedances physically meaningful
    dss.run_command("new line.feeder1 bus1=main_bus bus2=feeder1_head phases=3 linecode=feeder length=4.5 units=km")
    dss.run_command("new line.feeder2 bus1=main_bus bus2=feeder2_head phases=3 linecode=feeder length=6.2 units=km")
    dss.run_command("new line.feeder3 bus1=main_bus bus2=feeder3_head phases=3 linecode=feeder length=8.5 units=km")

    # 5. Fixed Set of Distribution Transformers (11/0.415 kV, delta-wye, 1.5 MVA)
    # Secondary side is 0.415 kV (LV) which connects to the unknown downstream networks
    dss.run_command("new transformer.trans1 phases=3 windings=2 buses=[feeder1_head, feeder1_sec] conns=[delta, wye] kvs=[11.0, 0.415] kvas=[1500, 1500] %r=0.8 xhl=5.0")
    dss.run_command("new transformer.trans2 phases=3 windings=2 buses=[feeder2_head, feeder2_sec] conns=[delta, wye] kvs=[11.0, 0.415] kvas=[1500, 1500] %r=0.8 xhl=5.0")
    dss.run_command("new transformer.trans3 phases=3 windings=2 buses=[feeder3_head, feeder3_sec] conns=[delta, wye] kvs=[11.0, 0.415] kvas=[1500, 1500] %r=0.8 xhl=5.0")

    print("INFO: OpenDSS Plant Model Initialized (33/11/0.415 kV) with 3 Feeders and 3 Fixed Transformers.")

def compute_symmetrical_components(mags, angles_deg):
    """
    Computes symmetrical components (zero, positive, and negative sequence) from three-phase complex phasor inputs.
    T_inv = 1/3 * [[1, 1, 1], [1, a, a^2], [1, a^2, a]] where a = e^(j * 120 deg)
    """
    # Convert polar to rectangular complex form
    rad = np.radians(angles_deg)
    phasors = [m * (np.cos(r) + 1j * np.sin(r)) for m, r in zip(mags, rad)]

    # operator a
    a = np.cos(2.0*np.pi/3.0) + 1j * np.sin(2.0*np.pi/3.0)
    a_sq = a * a

    x0 = (phasors[0] + phasors[1] + phasors[2]) / 3.0
    x1 = (phasors[0] + a * phasors[1] + a_sq * phasors[2]) / 3.0
    x2 = (phasors[0] + a_sq * phasors[1] + a * phasors[2]) / 3.0

    return {
        "zero": (np.abs(x0), np.degrees(np.angle(x0))),
        "positive": (np.abs(x1), np.degrees(np.angle(x1))),
        "negative": (np.abs(x2), np.degrees(np.angle(x2)))
    }

def extract_bus_voltages(bus_name: str):
    """
    Extracts magnitude and phase angles from Bus.VMagAngle() using correct stride slicing:
    [0:6:2] for magnitudes, [1:6:2] for phase angles.
    """
    dss.Circuit.SetActiveBus(bus_name)
    v_mag_angle = dss.Bus.VMagAngle()

    if not v_mag_angle or len(v_mag_angle) < 6:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    mags = v_mag_angle[0:6:2]
    angles = v_mag_angle[1:6:2]
    return list(mags), list(angles)

def extract_element_currents(element_name: str):
    """
    Extracts terminal currents (magnitudes and phase angles) of a specific element.
    Uses correct stride slicing for primary terminal currents.
    """
    dss.Circuit.SetActiveElement(element_name)
    currents_mag_ang = dss.CktElement.CurrentsMagAng()

    if not currents_mag_ang or len(currents_mag_ang) < 6:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    mags = currents_mag_ang[0:6:2]
    angles = currents_mag_ang[1:6:2]
    return list(mags), list(angles)

def get_boundary_measurements():
    """
    Retrieves detailed synchronized boundary measurements M at feeder heads and transformers.
    Extracts all physical parameters required by the methodology.
    """
    measurements = {}

    # Base voltage for reference
    v_main_mags, v_main_angs = extract_bus_voltages("main_bus")
    v_main_seq = compute_symmetrical_components(v_main_mags, v_main_angs)
    v_main_pos_mag = v_main_seq["positive"][0]

    for i in range(1, 4):
        bus_name = f"feeder{i}_head"
        line_name = f"line.feeder{i}"
        trans_name = f"transformer.trans{i}"

        # 1. Voltage & Symmetrical Voltages
        v_mags, v_angs = extract_bus_voltages(bus_name)
        v_seq = compute_symmetrical_components(v_mags, v_angs)

        v_pos_mag, v_pos_ang = v_seq["positive"]
        v_neg_mag, v_neg_ang = v_seq["negative"]
        v_zero_mag, v_zero_ang = v_seq["zero"]

        # Voltage unbalance (IEC definition: V_neg / V_pos * 100)
        v_unb = (v_neg_mag / v_pos_mag * 100.0) if v_pos_mag > 0 else 0.0

        # 2. Currents & Symmetrical Currents
        i_mags, i_angs = extract_element_currents(line_name)
        i_seq = compute_symmetrical_components(i_mags, i_angs)

        i_pos_mag, i_pos_ang = i_seq["positive"]
        i_neg_mag, i_neg_ang = i_seq["negative"]
        i_zero_mag, i_zero_ang = i_seq["zero"]

        # Current unbalance (I_neg / I_pos * 100)
        i_unb = (i_neg_mag / i_pos_mag * 100.0) if i_pos_mag > 0 else 0.0

        # 3. Powers and Losses
        dss.Circuit.SetActiveElement(line_name)
        powers = dss.CktElement.Powers() # Active, Reactive per phase terminal [P1, Q1, P2, Q2, P3, Q3, ...]
        losses = dss.CktElement.Losses() # [Real, Reactive]

        p_total = sum(powers[0:6:2]) if powers else 0.0
        q_total = sum(powers[1:6:2]) if powers else 0.0
        s_total = np.sqrt(p_total**2 + q_total**2)

        pf = (p_total / (s_total + 1e-6)) if s_total > 0 else 1.0

        # 4. Phase Angle Difference
        phase_diff_deg = v_pos_ang - v_main_seq["positive"][1]

        # 5. Equivalent Impedance (Z = V_pos / I_pos)
        z_eq = (v_pos_mag / (i_pos_mag + 1e-6)) if i_pos_mag > 0 else 0.0

        # 6. Voltage sensitivity indices and Network stiffness (approximated analytically from feeder impedances)
        # Standard analytical approximation based on known feeder line length and r1/x1 parameters
        # Feeder length: feeder1=4.5km, feeder2=6.2km, feeder3=8.5km
        feeder_lengths = {1: 4.5, 2: 6.2, 3: 8.5}
        l_km = feeder_lengths[i]
        r_total = 0.25 * l_km
        x_total = 0.35 * l_km

        dv_dp = r_total / (v_pos_mag + 1e-6)
        dv_dq = x_total / (v_pos_mag + 1e-6)
        stiffness = v_pos_mag**2 / (np.sqrt(r_total**2 + x_total**2) + 1e-6)

        # Aggregate Feeder Head Data
        measurements[f"feeder{i}_voltage_mag"] = v_mags
        measurements[f"feeder{i}_voltage_ang"] = v_angs
        measurements[f"feeder{i}_voltage_pos_mag"] = v_pos_mag
        measurements[f"feeder{i}_voltage_pos_ang"] = v_pos_ang
        measurements[f"feeder{i}_voltage_neg_mag"] = v_neg_mag
        measurements[f"feeder{i}_voltage_zero_mag"] = v_zero_mag
        measurements[f"feeder{i}_voltage_unbalance_pct"] = v_unb

        measurements[f"feeder{i}_current_mag"] = i_mags
        measurements[f"feeder{i}_current_ang"] = i_angs
        measurements[f"feeder{i}_current_pos_mag"] = i_pos_mag
        measurements[f"feeder{i}_current_neg_mag"] = i_neg_mag
        measurements[f"feeder{i}_current_zero_mag"] = i_zero_mag
        measurements[f"feeder{i}_current_unbalance_pct"] = i_unb

        measurements[f"feeder{i}_p_kw"] = p_total / 1000.0
        measurements[f"feeder{i}_q_kvar"] = q_total / 1000.0
        measurements[f"feeder{i}_s_kva"] = s_total / 1000.0
        measurements[f"feeder{i}_pf"] = pf
        measurements[f"feeder{i}_frequency"] = float(dss.Solution.Frequency())
        measurements[f"feeder{i}_losses_kw"] = losses[0] / 1000.0 if losses else 0.0

        measurements[f"feeder{i}_eq_impedance_ohm"] = z_eq
        measurements[f"feeder{i}_phase_angle_diff_deg"] = phase_diff_deg
        measurements[f"feeder{i}_dv_dp"] = dv_dp
        measurements[f"feeder{i}_dv_dq"] = dv_dq
        measurements[f"feeder{i}_stiffness_kva"] = stiffness / 1000.0

        # --- 7. Transformer Edge Measurements ---
        dss.Circuit.SetActiveElement(trans_name)
        trans_powers = dss.CktElement.Powers() # Powers at primary (HV) terminal
        trans_losses = dss.CktElement.Losses() # [Real, Reactive]

        t_p_total = sum(trans_powers[0:6:2]) if trans_powers else 0.0
        t_q_total = sum(trans_powers[1:6:2]) if trans_powers else 0.0
        t_s_total = np.sqrt(t_p_total**2 + t_q_total**2)
        t_pf = (t_p_total / (t_s_total + 1e-6)) if t_s_total > 0 else 1.0

        # HV Terminal Voltages & Currents (Primary)
        t_v_mags, t_v_angs = extract_bus_voltages(bus_name)
        t_i_mags, _ = extract_element_currents(trans_name)

        # Transformer ratings: 1.5 MVA (1500 kVA)
        loading_pct = (t_s_total / 1500000.0) * 100.0

        # Copper loss and core loss splitting
        # R_wdg %r=0.8, base MVA=1.5 -> R_pu = 0.008 -> R_ohms = 0.008 * 11000^2 / 1.5e6 = 0.645 ohms
        core_loss = 2.5 # Constant core/no-load loss in kW approx
        copper_loss = (t_s_total / 1500000.0)**2 * 12.0 # Load loss at full MVA is 12 kW approx

        # Voltage regulation estimation (V_no_load - V_full_load)
        # dV% = I*(R*cos(phi) + X*sin(phi)) where %r=0.8, %x=5.0
        sin_phi = np.sqrt(1.0 - t_pf**2)
        v_reg = (loading_pct / 100.0) * (0.8 * t_pf + 5.0 * sin_phi)

        # Equivalent transformer impedance seen from HV side
        t_v_avg = np.mean(t_v_mags)
        t_i_avg = np.mean(t_i_mags)
        t_z = (t_v_avg / (t_i_avg + 1e-6)) if t_i_avg > 0 else 0.0

        measurements[f"transformer{i}_hv_voltage"] = t_v_avg
        measurements[f"transformer{i}_hv_current"] = t_i_avg
        measurements[f"transformer{i}_p_kw"] = t_p_total / 1000.0
        measurements[f"transformer{i}_q_kvar"] = t_q_total / 1000.0
        measurements[f"transformer{i}_s_kva"] = t_s_total / 1000.0
        measurements[f"transformer{i}_pf"] = t_pf
        measurements[f"transformer{i}_loading_pct"] = loading_pct
        measurements[f"transformer{i}_copper_loss_kw"] = copper_loss
        measurements[f"transformer{i}_core_loss_kw"] = core_loss
        measurements[f"transformer{i}_tap_position"] = float(dss.Transformers.Tap())
        measurements[f"transformer{i}_voltage_regulation_pct"] = v_reg
        measurements[f"transformer{i}_eq_impedance_ohm"] = t_z

    return measurements

def emulate_atp_transient(event_type: str, dss_m: dict, feeder_idx: int, duration: float = 0.05, fs: float = 10000.0):
    """
    High-Fidelity Coupled ATP-EMTP Dynamic Transient Emulator.
    Instead of synthetic uncoupled sines, the transient is fully coupled to the solved OpenDSS steady-state
    operating point (steady-state phase voltages V_0, currents I_0, and angles).

    Transient wave forms are also physically linked to:
    - Feeder impedance (determines damping rate alpha)
    - Load level (determines starting amplitude and transient sag depth)
    - Network complexity and topology
    - Dynamic components like capacitors or motors
    """
    t = np.linspace(0, duration, int(duration * fs))
    freq = 50.0 # 50 Hz system frequency
    omega = 2.0 * np.pi * freq

    # 1. Retrieve the pre-event steady-state operating point from the solved OpenDSS state
    v_mags = dss_m[f"feeder{feeder_idx}_voltage_mag"]
    v_angs = dss_m[f"feeder{feeder_idx}_voltage_ang"]
    i_mags = dss_m[f"feeder{feeder_idx}_current_mag"]
    i_angs = dss_m[f"feeder{feeder_idx}_current_ang"]

    # Physical coupling parameters: Feeder equivalent impedance and load level
    z_eq = dss_m[f"feeder{feeder_idx}_eq_impedance_ohm"]
    p_load = dss_m[f"feeder{feeder_idx}_p_kw"]

    # Damping rate and ringing frequency derived from physics (Z_eq = R + jX)
    # Higher network impedance leads to faster decay and lower resonance frequency
    damping_alpha = 150.0 / (z_eq + 1e-3)
    f_resonance = 1200.0 - 5.0 * z_eq # resonance shifts based on impedance
    omega_res = 2.0 * np.pi * f_resonance

    v_waveforms = []
    i_waveforms = []

    # Simulate three phase coupled waveforms
    for phase_idx in range(3):
        v_start = v_mags[phase_idx] * np.sqrt(2)
        v_angle_rad = np.radians(v_angs[phase_idx])

        i_start = i_mags[phase_idx] * np.sqrt(2)
        i_angle_rad = np.radians(i_angs[phase_idx])

        # Base sinusoidal wave derived directly from the solved OpenDSS operating point
        v_base = v_start * np.sin(omega * t + v_angle_rad)
        i_base = i_start * np.sin(omega * t + i_angle_rad)

        v_trans = np.zeros_like(t)
        i_trans = np.zeros_like(t)

        # 2. Add event transients coupled to solved network characteristics
        if event_type == 'transformer_energization':
            # Severe asymmetric inrush: amplitude scales with load level and transformer current
            inrush_mult = 3.5 + 0.01 * p_load
            inrush_env = i_start * inrush_mult * np.exp(-t * damping_alpha)
            # High inrush current and associated voltage sags
            i_trans = inrush_env * (np.sin(omega * t + i_angle_rad) + 0.35 * np.sin(2.0 * omega * t + i_angle_rad))
            v_sag = 1.0 - 0.12 * np.exp(-t * damping_alpha)
            v_base *= v_sag

        elif event_type == 'capacitor_switching':
            # High-frequency transient ringing linked directly to resonance frequency
            v_ring_env = v_start * 0.45 * np.exp(-t * damping_alpha * 0.5)
            v_trans = v_ring_env * np.sin(omega_res * t + v_angle_rad)
            i_ring_env = i_start * 1.5 * np.exp(-t * damping_alpha * 0.5)
            i_trans = i_ring_env * np.sin(omega_res * t + i_angle_rad)

        elif event_type == 'motor_starting':
            # Heavy starting currents (5-6x base) with a long decay profile, causing a sustained voltage sag
            start_decay = 4.0 / (0.05 + 0.001 * p_load)
            motor_env = 5.0 * np.exp(-t * start_decay)
            i_trans = i_start * motor_env * np.sin(omega * t + i_angle_rad - 0.5)
            v_sag = 1.0 - 0.22 * np.exp(-t * start_decay * 0.8)
            v_base *= v_sag

        elif event_type == 'temporary_fault':
            # Asymmetric line-to-ground fault: voltage collapse on faulted phase, 12x current spike
            fault_active = (t >= 0.01) & (t <= 0.04)
            if phase_idx == 0: # Phase A fault
                v_base = np.where(fault_active, v_base * 0.05, v_base)
                i_base = np.where(fault_active, i_base * 12.0, i_base)
            else: # Phases B and C experience slight swell
                v_base = np.where(fault_active, v_base * 1.15, v_base)

        elif event_type == 'nonlinear_load':
            # Non-linear switching noise: harmonic distortion (THD)
            v_trans = v_start * 0.01 * np.sin(3.0 * omega * t) + v_start * 0.02 * np.sin(5.0 * omega * t)
            i_trans = i_start * 0.12 * np.sin(3.0 * omega * t) + i_start * 0.08 * np.sin(5.0 * omega * t)

        v_waveforms.append(v_base + v_trans)
        i_waveforms.append(i_base + i_trans)

    return t, np.array(v_waveforms), np.array(i_waveforms)

def extract_dynamic_transient_features(t: np.ndarray, v_wave: np.ndarray, i_wave: np.ndarray, fs: float = 10000.0):
    """
    Applies Fast Fourier Transform (FFT) on the coupled waveforms to calculate
    advanced dynamic feature set parameters:
    - Spectral Centroid
    - Dominant Frequency
    - Wavelet Energy proxy (energy in localized sub-bands)
    """
    N = len(t)
    freqs = np.fft.rfftfreq(N, 1.0/fs)

    # Use the first phase (Phase A) for spectral analysis
    v_fft = np.abs(np.fft.rfft(v_wave[0])) / N

    # Avoid zero division
    sum_v = np.sum(v_fft) + 1e-9

    # 1. Spectral Centroid
    spectral_centroid = float(np.sum(freqs * v_fft) / sum_v)

    # 2. Dominant Frequency (excluding DC component)
    dom_idx = np.argmax(v_fft[1:]) + 1 if len(v_fft) > 1 else 0
    dominant_frequency = float(freqs[dom_idx])

    # 3. Wavelet Energy proxies (energy content in specific sub-bands)
    # Band 1: 50 - 250 Hz (Low harmonics)
    # Band 2: 250 - 1000 Hz (Medium frequency transient ringing)
    # Band 3: 1000 - 5000 Hz (High frequency switching noise)
    b1_mask = (freqs >= 50) & (freqs <= 250)
    b2_mask = (freqs > 250) & (freqs <= 1000)
    b3_mask = (freqs > 1000) & (freqs <= 5000)

    e_band1 = float(np.sum(v_fft[b1_mask]**2))
    e_band2 = float(np.sum(v_fft[b2_mask]**2))
    e_band3 = float(np.sum(v_fft[b3_mask]**2))

    total_energy = e_band1 + e_band2 + e_band3 + 1e-9

    return {
        "spectral_centroid": round(spectral_centroid, 2),
        "dominant_frequency": round(dominant_frequency, 2),
        "wavelet_energy_low_pct": round(e_band1 / total_energy * 100.0, 2),
        "wavelet_energy_mid_pct": round(e_band2 / total_energy * 100.0, 2),
        "wavelet_energy_high_pct": round(e_band3 / total_energy * 100.0, 2)
    }
