%% Robust BMS Validation Report (Clean Decomposition Focus)
% This script verifies the BMS Layer (State Estimation, Protection, and Safety).
% Ref: docs/paper.md Section 2.

%% 1. Initialization & Configuration
% Mock parameters representing a fixed cell plant model
params = struct(...
    'Nominal_Cap_Ah', 10, ...
    'R_nominal', 0.01, ...
    'V_max', 3.8, ...
    'V_min', 2.0, ...
    'T_max', 65 ...
);

%% 2. State Estimation Verification (SOC & SOH)
fprintf('--- Testing State Estimation ---\n');
inputs = struct('V_cells', [3.3, 3.3], 'T_cells', [25, 25], 'I_measured', 0, ...
                'Mode', 'Standby', 'Fault_Reset', 0, 'I_request', 0);

% Test SOH Inference (Impedance growth detection)
inputs.I_measured = 10; % Discharge
inputs.V_cells = [3.1, 3.1]; % High voltage drop -> High resistance
[~, states] = bms_control_logic(inputs, params);
fprintf('  SOH (Resistance-based): %.2f%%\n', min(states.SOH_R)*100);

%% 3. Protection Logic Verification (OV/UV/OT)
fprintf('\n--- Testing Protection Logic ---\n');

% Over-Temperature (OT)
inputs.T_cells = [70, 25];
[I_cmd, states] = bms_control_logic(inputs, params);
fprintf('  OT Trigger: State=%s, I_cmd=%.1f\n', states.bms_state, I_cmd);

% Fault Reset
inputs.T_cells = [25, 25];
inputs.Fault_Reset = 1;
[~, states] = bms_control_logic(inputs, params);
fprintf('  Fault Reset: State=%s\n', states.bms_state);

% Over-Voltage (OV)
inputs.Fault_Reset = 0;
inputs.V_cells = [3.9, 3.3];
[I_cmd, states] = bms_control_logic(inputs, params);
fprintf('  OV Trigger: State=%s, I_cmd=%.1f\n', states.bms_state, I_cmd);

%% 4. Safety Enforcement (Thermal & SOC Derating)
fprintf('\n--- Testing Safety Enforcement & Derating ---\n');
clear bms_control_logic; % Reset persistent states
inputs.Mode = 'Run';
inputs.V_cells = [3.3, 3.3];
inputs.T_cells = [58, 58]; % Near T_max (65)
inputs.I_request = 10;
[I_cmd, ~] = bms_control_logic(inputs, params);
fprintf('  Thermal Derating: Req=10.0, Cmd=%.2f\n', I_cmd);

% Low SOC Derating
clear bms_control_logic;
inputs.T_cells = [25, 25];
% Mock internal SOC state being low (BMS estimates it)
% In ecu.m, SOC is persistent and updated. We'll simulate a few steps.
for i = 1:5
    inputs.I_measured = 50; % High discharge
    inputs.V_cells = [2.2, 2.2]; % Low voltage -> Low SOC
    [I_cmd, states] = bms_control_logic(inputs, params);
end
fprintf('  Low SOC Derating: SOC=%.3f, I_cmd=%.2f\n', states.SOC(1), I_cmd);

%% 5. Control Logic (Cell Balancing)
fprintf('\n--- Testing Cell Balancing ---\n');
inputs.I_measured = 0; % Idle
inputs.V_cells = [3.4, 3.3]; % Imbalance
[~, states] = bms_control_logic(inputs, params);
fprintf('  Balancing Active: [%d, %d]\n', states.balancing_active(1), states.balancing_active(2));

%% Summary
fprintf('\nBMS Layer Validation Complete.\n');
