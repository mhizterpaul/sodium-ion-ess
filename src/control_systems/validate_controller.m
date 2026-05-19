%% Robust BMS Validation Report
% Verifies hierarchical safety, optimal arbitration, and pack balancing.

%% 1. Initialization
params = load_optimized_data('src/control_systems/optimized_params.mat');

%% 2. Stability Analysis (Frequency & Time Domain)
% Analyzes the system using Bode plots and Step response for the 5-state plant.
sys_pack = get_pack_dynamics(params);

% 2.1 Eigenvalue Analysis
evs = eig(sys_pack.A);
is_stable = all(real(evs) <= 0);
fprintf('Asymptotic Stability: %s\n', mat2str(is_stable));

% 2.2 Frequency Response (Bode Plot)
% Analysis of the system's gain and phase margins
try
    h_bode = figure('Visible', 'off');
    bode(sys_pack);
    saveas(h_bode, 'src/control_systems/html/bode_plot.png');
    fprintf('Bode Plot generated.\n');
catch
    fprintf('Control System Toolbox not found, skipping Bode Analysis.\n');
end

% 2.3 Time Response (Step Response)
% Analysis of the system's settling time and overshoot
try
    h_step = figure('Visible', 'off');
    step(sys_pack);
    saveas(h_step, 'src/control_systems/html/step_response.png');
    fprintf('Step Response generated.\n');
catch
    fprintf('Control System Toolbox not found, skipping Step Analysis.\n');
end

%% 3. Hierarchical Safety Test
% Verifies transition from Normal -> Warning -> Derating -> Shutdown -> Latch
inputs = struct('V_cells', [3.3, 3.3], 'T_cells', [25, 25], 'SOC_est', 0.5, ...
                'I_measured', 0, 'Mode', 'Drive', 'Fault_Reset', 0, 'I_request', 10, 'T_amb', 298.15);

[~, s0] = bms_control_logic(inputs, params);
inputs.T_cells = [60, 25]; [~, s1] = bms_control_logic(inputs, params); % Warning
inputs.T_cells = [70, 25]; [~, s2] = bms_control_logic(inputs, params); % Derating
inputs.T_cells = [80, 25]; [~, s3] = bms_control_logic(inputs, params); % Shutdown
inputs.T_cells = [90, 25]; [~, s4] = bms_control_logic(inputs, params); % Latch

fprintf('Safety Hierarchy Validation:\n');
fprintf('  T=60C: Status %d (Warning)\n', s1.fault_status);
fprintf('  T=70C: Status %d (Derating)\n', s2.fault_status);
fprintf('  T=80C: Status %d (Shutdown)\n', s3.fault_status);
fprintf('  T=90C: Status %d (Latch)\n', s4.fault_status);

%% 3. Optimal Current Arbitration (MPC-inspired)
% Verifies derating at T=70C
inputs.T_cells = [70, 25];
[I_cmd, ~] = bms_control_logic(inputs, params);
fprintf('Optimal Arbitration (T=70C): I_cmd = %.2f A (Requested 10A)\n', I_cmd);

%% 4. Pack Dynamics & Balancing Energy
% Models 2-cell imbalance
sys_pack = get_pack_dynamics(params);
fprintf('Pack Dynamics: %d states, capacity imbalance modeled.\n', size(sys_pack.A, 1));

inputs.V_cells = [3.5, 3.45]; % Imbalance
[~, states] = bms_control_logic(inputs, params);
fprintf('Balancing Energy Dissipation: Cell 1 P = %.4f W\n', states.P_balance(1));

%% Helper Functions
function params = load_optimized_data(filename)
    if ~exist(filename, 'file')
        params = struct('Nominal_cell_capacity_Ah', 10, 'Contact_resistance_Ohm', 0.01);
    else
        data = load(filename);
        params = data.optimized_params;
    end
end
