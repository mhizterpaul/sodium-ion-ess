%% SIMULINK BMS CONTROL SYSTEM MODEL
% Ref: docs/paper.md

function [I_cmd] = bms_control_logic(SOC, V_t, T, SOH, Grid_Stress, params)
    % params: Optimized cell parameters struct (consumed from JSON)

    Q_n = params.capacity_ah;
    T_safe = 85; % Celsius
    lambda = 0.5;

    %% 3. C-RATE CONTROLLER
    C_ref = 1.0;
    I_ref = C_ref * Q_n;

    %% 5. THERMAL LIMITER
    % I_cmd_th = I_cmd * exp(-lambda * (T - T_safe)^+)
    if T > T_safe
        thermal_scaling = exp(-lambda * (T - T_safe));
    else
        thermal_scaling = 1.0;
    end

    if T > 85
        I_thermal = 0;
    else
        I_thermal = I_ref * thermal_scaling;
    end

    %% 6. GRID STRESS DERATING
    % Stress metric D_k = alpha*|dV| + beta*|df| + gamma*B
    % Current scaling I_cmd_grid = I_cmd * exp(-mu * D_k)
    mu = 0.2;
    grid_scaling = exp(-mu * Grid_Stress);
    I_grid = I_ref * grid_scaling;

    %% Current Arbitration
    % I = min(I_C_rate, I_thermal, I_grid, I_SOH)
    I_cmd = min([I_ref, I_thermal, I_grid]);

end

function load_optimized_params(filename)
    % Reads JSON exported from optimization pipeline
    val = jsondecode(fileread(filename));
    disp('Consuming Optimized Cell Parameters:');
    disp(val);
end
