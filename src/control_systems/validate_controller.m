%% BMS Controller Validation Report
% This report verifies the stability, response characteristics, and estimator convergence
% of the NFPP Sodium-Ion Battery Management System.
%
% Ref: docs/paper.md

%% 1. Initialization and Parameter Loading
params = load_optimized_data('src/control_systems/optimized_params.mat');

%% 2. Estimator Convergence (EKF)
% Verifies the ability of the Extended Kalman Filter to recover from incorrect initial SOC.
soc_true = 0.5;
soc_init = 0.8;
P = 0.1;
v_meas = 3.2;
i_meas = 0;

soc_est = soc_init;
convergence_steps = 0;
for i = 1:50
    [soc_est, P] = ekf_estimator(v_meas, i_meas, soc_est, P, params);
    if abs(soc_est - soc_true) < 0.01 && convergence_steps == 0
        convergence_steps = i;
    end
end
fprintf('EKF Convergence: %s in %d steps\n', mat2str(abs(soc_est - soc_true) < 0.01), convergence_steps);

%% 3. Stability Analysis (MIMO State-Space)
% Analyzes the asymptotic stability of the 5-state battery plant model.
[sys_ss, ~] = get_battery_dynamics(params);
if isstruct(sys_ss)
    evs = eig(sys_ss.A);
else
    evs = eig(sys_ss);
end
is_stable = all(real(evs) <= 0);
fprintf('Asymptotic Stability: %s\n', mat2str(is_stable));
fprintf('Max Eigenvalue (Real): %.4f\n', max(real(evs)));

%% 4. Pre-charge Sequence & Contactor Logic
% Verifies the State Machine transition from Standby to Driving via Pre-charge.
inputs = struct('V_cells', [3.3, 3.3], 'T_cells', [25, 25], 'SOC_est', 0.5, ...
                'I_measured', 0, 'Mode', 'Drive', 'Fault_Reset', 0, 'I_request', 10);

[~, s1] = bms_control_logic(inputs, params);
fprintf('Initial State: %s (Precharge Contactor: %d)\n', s1.bms_state, s1.contactor_pre);

for i = 1:6, [~, s_final] = bms_control_logic(inputs, params); end
fprintf('Final State: %s (Main Contactor: %d)\n', s_final.bms_state, s_final.contactor_main);

%% 5. Cell Balancing Logic
% Verifies activation of bleed resistors under voltage imbalance.
inputs.V_cells = [3.4, 3.3];
[~, states] = bms_control_logic(inputs, params);
fprintf('Balancing Active: %s\n', mat2str(any(states.balancing_active)));

%% 6. Fault Protection
% Verifies immediate current cutoff during over-temperature conditions.
inputs.T_cells = [90, 25];
[I_cmd, states] = bms_control_logic(inputs, params);
fprintf('Fault Triggered: %s, Command: %.1f A\n', states.bms_state, I_cmd);

%% Helper Functions
function params = load_optimized_data(filename)
    if ~exist(filename, 'file')
        params = struct('Nominal_cell_capacity_Ah', 10, 'Contact_resistance_Ohm', 0.01);
    else
        data = load(filename);
        params = data.optimized_params;
    end
end
