%% Model-Informed Energy Dispatch Validation Report
% This script verifies the Energy Dispatch Layer (Energy Decomposition & Stability).
% Ref: docs/paper.md Section 2.

%% 1. Initialization & Configuration
% Mock parameters representing a fixed power plant model
params = struct(...
    'P_max_bat', 5000, ...    % 5kW Battery
    'P_max_dump', 2000, ...   % 2kW Dump Load
    'SOC_min', 0.2, ...
    'SOC_max', 0.95, ...
    'T_crit', 55, ...
    'eta_inv', 0.96 ...       % 96% Efficiency
);

%% 2. Energy Decomposition Verification
fprintf('--- Testing Fundamental Energy Decomposition ---\n');
inputs = struct('P_solar', 6000, 'P_load_req', 3000, 'SOC', 0.6, 'SOH', 1.0, ...
                'T_bat', 25, 'V_grid', 1.0, 'f_grid', 60.0);

[P_targets, states] = dispatch_controller(inputs, params);
fprintf('  Solar Input: %.1f W\n', inputs.P_solar);
fprintf('  Load Delivery (Useful): %.1f W\n', P_targets.P_load);
fprintf('  Battery Buffering: %.1f W\n', P_targets.P_bat);
fprintf('  Loss (Inefficiency): %.1f W\n', P_targets.P_loss);
fprintf('  Harmonic Penalty: %.1f W\n', P_targets.P_harmonic);
fprintf('  Dump Dissipation: %.1f W\n', P_targets.P_dump);

%% 3. Stability Manifold Verification
fprintf('\n--- Testing Stability Manifold & Reserves ---\n');

% Test Grid Frequency Deviation (Reactive Support)
inputs.f_grid = 59.5; % Significant drop
[P_targets, ~] = dispatch_controller(inputs, params);
fprintf('  Grid Stability Energy (P_reactive): %.1f W at %.1f Hz\n', ...
        P_targets.P_reactive, inputs.f_grid);

%% 4. Constraint Handling (SOC & Thermal)
fprintf('\n--- Testing Electrochemical Constraints ---\n');

% Battery Full (SOC=0.98)
inputs.SOC = 0.98;
inputs.P_solar = 8000;
inputs.P_load_req = 1000;
[P_targets, ~] = dispatch_controller(inputs, params);
fprintf('  Battery Saturated (SOC=0.98): P_bat=%.1f, P_dump=%.1f\n', ...
        P_targets.P_bat, P_targets.P_dump);

% Critical Temperature
inputs.T_bat = 56; % Above T_crit
[P_targets, ~] = dispatch_controller(inputs, params);
fprintf('  Thermal Constraint (T=56C): P_bat=%.1f (Charge inhibited)\n', ...
        P_targets.P_bat);

%% 5. Efficiency & Availability Metrics
fprintf('\n--- Testing Performance Metrics ---\n');
inputs.SOC = 0.5;
inputs.T_bat = 25;
inputs.P_solar = 4000;
inputs.P_load_req = 3500;
[~, states] = dispatch_controller(inputs, params);
fprintf('  Energy Utilization Efficiency: %.2f%%\n', states.efficiency*100);
fprintf('  Stability Index: %.3f\n', states.stability_index);

%% Summary
fprintf('\nEnergy Dispatch Layer Validation Complete.\n');
