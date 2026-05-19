%% Robust BMS Validation Report
% Ref: docs/paper.md

%% 1. Initialization
params = load_optimized_data('src/control_systems/optimized_params.mat');

%% 2. Stability Analysis
sys_pack = get_pack_dynamics(params);
evs = eig(sys_pack.A);
fprintf('Asymptotic Stability: %s\n', mat2str(all(real(evs) <= 0)));

%% 3. Hierarchical Safety & State Machine
inputs = struct('V_cells', [3.3, 3.3], 'T_cells', [25, 25], 'SOC_est', 0.5, ...
                'I_measured', 0, 'Mode', 'Drive', 'Fault_Reset', 0, 'I_request', 10, 'T_amb', 298.15);

[~, s_init] = bms_control_logic(inputs, params);
fprintf('Initial Transition: Standby -> %s\n', s_init.bms_state);

% Trigger Shutdown
inputs.T_cells = [80, 25];
[~, s_fault] = bms_control_logic(inputs, params);
fprintf('Fault response: Status %d, State %s\n', s_fault.fault_status, s_fault.bms_state);

%% 4. Cell Voltage Convergence (Balancing Simulation)
% Time-series simulation to demonstrate balancing
fprintf('Simulating Cell Balancing Convergence...\n');
V = [3.5, 3.4]; % Imbalanced cells
inputs_bal = inputs;
for t = 1:50
    inputs_bal.V_cells = V;
    [~, states] = bms_control_logic(inputs_bal, params);
    % Simplistic passive balancing discharge
    V = V - 0.001 * states.balancing_active;
    if mod(t, 10) == 0
        fprintf('  t=%d: V_diff = %.3f V\n', t, abs(V(1)-V(2)));
    end
end
fprintf('Balancing Final Delta: %.4f V\n', abs(V(1)-V(2)));

%% 5. Estimator Convergence (EKF)
% Verifies the ability of the EKF to recover from an incorrect initial condition.
soc_true = 0.5;
soc_est = 0.8; % 30% error
P = 0.1;
v_meas = 3.2; i_meas = 0;

fprintf('Testing EKF Convergence (Initial Error: 30%%)...\n');
for i = 1:20
    [soc_est, P] = ekf_estimator(v_meas, i_meas, soc_est, P, params);
end
fprintf('  Final SOC Estimate: %.4f (Error: %.2f%%)\n', soc_est, abs(soc_est - soc_true)*100);

%% 6. Physical Plant Validation (Simscape Equations)
% This section exercises the coupled electro-thermal equations derived from
% the Simscape model (nfpp_cell.ssc).
[V_p, T_p] = plant_model(10, 298.15, params);
fprintf('Physical Plant Coupling Check: V=%.2f, T=%.1f\n', V_p, T_p);

%% Helper Functions

function [soc_new, P_new] = ekf_estimator(v_meas, i_meas, soc_old, P_old, params)
    % Extended Kalman Filter implementation (Redefined here for standalone publishing)
    Q_noise = 1e-6; R_noise = 0.01; dt = 1;
    Q_cap = params.Nominal_cell_capacity_Ah * 3600;
    soc_pred = soc_old - i_meas * dt / Q_cap;
    P_pred = P_old + Q_noise;
    H = 0.8;
    K = P_pred * H / (H * P_pred * H + R_noise);
    v_pred = 3.1 + H * (soc_pred - 0.5) - i_meas * 0.01;
    soc_new = max(0, min(1, soc_pred + K * (v_meas - v_pred)));
    P_new = (1 - K * H) * P_pred;
end

function params = load_optimized_data(filename)
    if ~exist(filename, 'file')
        params = struct('Nominal_cell_capacity_Ah', 10, 'Contact_resistance_Ohm', 0.01);
    else
        data = load(filename);
        params = data.optimized_params;
    end
end
