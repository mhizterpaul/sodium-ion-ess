%% SIMULINK BMS CONTROL SYSTEM MODEL (ECU Logic)
% Ref: docs/paper.md

function [I_cmd, states] = bms_control_logic(inputs, params)
    % inputs: struct {SOC_est, V_cells, T_cells, I_measured, Mode, Fault_Reset, I_request}

    persistent bms_state;
    persistent precharge_timer;
    if isempty(bms_state)
        bms_state = 'Standby';
        precharge_timer = 0;
    end

    %% 1. STATE MACHINE with Pre-charge Logic
    % Fault Monitoring (Continuous)
    is_fault = any(inputs.V_cells > 3.8) || any(inputs.V_cells < 2.0) || any(inputs.T_cells > 85);
    if is_fault
        bms_state = 'Fault';
    end

    switch bms_state
        case 'Standby'
            if inputs.Fault_Reset, bms_state = 'Standby'; end % Clear latch
            if strcmp(inputs.Mode, 'Drive') || strcmp(inputs.Mode, 'Charge')
                bms_state = 'Precharge';
                precharge_timer = 0;
            end
        case 'Precharge'
            % Simulate pre-charge contactor closing (special sequence)
            precharge_timer = precharge_timer + 1;
            if precharge_timer > 5 % 5 time steps for pre-charge
                if strcmp(inputs.Mode, 'Drive')
                    bms_state = 'Driving';
                else
                    bms_state = 'Charging';
                end
            end
        case 'Driving'
            if strcmp(inputs.Mode, 'Standby'), bms_state = 'Standby'; end
        case 'Charging'
            if strcmp(inputs.Mode, 'Standby') || all(inputs.SOC_est > 0.99), bms_state = 'Standby'; end
        case 'Fault'
            if inputs.Fault_Reset, bms_state = 'Standby'; end
    end

    %% 2. MONITORING AND PROTECTION
    Q_n = params.Nominal_cell_capacity_Ah;
    I_max_thermal = Q_n * exp(-0.5 * (max(inputs.T_cells) - 25)/20);

    %% 3. COMMAND CALCULATION
    if strcmp(bms_state, 'Driving')
        I_cmd = inputs.I_request;
    elseif strcmp(bms_state, 'Charging')
        I_cmd = Q_n * 0.5;
    elseif strcmp(bms_state, 'Precharge')
        I_cmd = 0.1 * Q_n; % Limited current during pre-charge
    else
        I_cmd = 0;
    end

    % Arbitration
    I_cmd = min(I_cmd, I_max_thermal);
    if strcmp(bms_state, 'Fault'), I_cmd = 0; end

    %% 4. CONTACTORS & BALANCING
    states.bms_state = bms_state;
    states.balancing_active = (inputs.V_cells - mean(inputs.V_cells)) > 0.01;
    states.I_limit = I_max_thermal;
    states.contactor_main = strcmp(bms_state, 'Driving') || strcmp(bms_state, 'Charging');
    states.contactor_pre = strcmp(bms_state, 'Precharge');
end

%% SOC ESTIMATOR (EKF)
function [soc_new, P_new] = ekf_estimator(v_meas, i_meas, soc_old, P_old, params)
    dt = 1;
    Q = params.Nominal_cell_capacity_Ah * 3600;
    R = params.Contact_resistance_Ohm + 0.01;

    soc_pred = soc_old - i_meas * dt / Q;
    P_pred = P_old + 1e-6;
    H = 0.5;
    K = P_pred * H / (H * P_pred * H + 0.01);
    v_pred = 3.2 + H * (soc_pred - 0.5) - i_meas * R;
    soc_new = soc_pred + K * (v_meas - v_pred);
    P_new = (1 - K * H) * P_pred;
end

%% PLANT MODEL PLACEHOLDERS (Simscape Equivalent)
function [V_out, T_out] = plant_model(I_in, T_amb, params)
    % Simple Equivalent Circuit + Thermal Layout
    R_int = params.Contact_resistance_Ohm + 0.01;
    C_th = 500; % Thermal mass
    V_out = 3.2 - I_in * R_int;
    T_out = T_amb + (I_in^2 * R_int) / C_th;
end
