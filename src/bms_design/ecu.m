%% ROBUST BMS LAYER ARCHITECTURE (Core Research Contribution)
% Focus: State Estimation, Protection Logic, Safety Enforcement, and Control Logic.
% This module implements the BMS ECU algorithms, assuming a fixed Cell Plant Model.

function [I_cmd, states] = bms_control_logic(inputs, params)
    % inputs: struct {V_cells, T_cells, I_measured, Mode, Fault_Reset, I_request}
    % params: struct {Nominal_Cap_Ah, R_nominal, V_max, V_min, T_max}

    persistent bms_state;
    persistent x_ekf;     % State vector for EKF: [SOC1, Vc1, SOC2, Vc2, ...]
    persistent P_ekf;     % Covariance matrix
    persistent R_est;     % Estimated internal resistance (for SOH/Protection)
    persistent fault_latched;

    num_cells = length(inputs.V_cells);
    dt = 0.1; % Step size

    if isempty(bms_state)
        bms_state = 'Standby';
        x_ekf = repmat([0.8; 0], num_cells, 1); % Initial SOC=0.8 and Vc=0
        P_ekf = eye(num_cells * 2) * 0.01;
        R_est = ones(num_cells, 1) * params.R_nominal;
        fault_latched = 0;
    end

    %% 1. STATE ESTIMATION (EKF-based SOC)
    Q_cap = params.Nominal_Cap_Ah * 3600;
    R0 = params.R_nominal;
    R1 = 0.005; C1 = 1000; % Mock RC parameters

    % Prediction and Update for each cell
    for i = 1:num_cells
        idx = (i-1)*2 + (1:2);
        soc = x_ekf(idx(1));
        vc = x_ekf(idx(2));

        % A. Predict
        soc_p = soc - inputs.I_measured * dt / Q_cap;
        vc_p = vc * exp(-dt/(R1*C1)) + inputs.I_measured * R1 * (1 - exp(-dt/(R1*C1)));
        x_ekf(idx) = [soc_p; vc_p];

        % B. Update
        % OCV(SOC) approximation
        v_oc = 2.0 + 3.5*soc_p - 5.1*soc_p^2 + 4.8*soc_p^3 - 2.1*soc_p^4 + 0.5*soc_p^5;
        v_pred = v_oc - inputs.I_measured * R0 - vc_p;

        H = [0.8, -1]; % dV/dSOC (avg) and dV/dVc
        P_sub = P_ekf(idx, idx);
        K = P_sub * H' / (H * P_sub * H' + 0.01);
        x_ekf(idx) = x_ekf(idx) + K * (inputs.V_cells(i) - v_pred);
        P_ekf(idx, idx) = (eye(2) - K * H) * P_sub;
    end

    % SOH Inference: Resistance estimation via RLS logic
    % R_est_new = (V_oc - V_meas - Vc) / I
    for i = 1:num_cells
        v_oc = 2.0 + 3.5*x_ekf((i-1)*2+1); % Simplified
        v_drop = abs(v_oc - inputs.V_cells(i) - x_ekf((i-1)*2+2));
        r_inst = v_drop / (abs(inputs.I_measured) + 1e-3);
        if abs(inputs.I_measured) > 1.0 % Only update when significant current flows
            R_est(i) = 0.99 * R_est(i) + 0.01 * r_inst;
        end
    end

    states.SOC = x_ekf(1:2:end);
    states.SOH_R = params.R_nominal ./ R_est;

    %% 2. PROTECTION LOGIC (Diagnostic & Safety)
    fault_code = 0;
    if any(inputs.V_cells > params.V_max), fault_code = 1; end
    if any(inputs.V_cells < params.V_min), fault_code = 1; end
    if any(inputs.T_cells > params.T_max), fault_code = 2; end
    if any(states.SOH_R < 0.5), fault_code = 3; end

    if fault_code > 0, fault_latched = 1; end
    if inputs.Fault_Reset, fault_latched = 0; end

    %% 3. SAFETY ENFORCEMENT & CURRENT ARBITRATION
    T_max_now = max(inputs.T_cells);
    thermal_derate = exp(-max(0, T_max_now - (params.T_max - 5)) / 2.0);
    soc_max_derate = min(1, max(0, (1.0 - states.SOC) / 0.05));
    soc_min_derate = min(1, max(0, (states.SOC - 0.0) / 0.05));

    if fault_latched
        I_cmd = 0;
    else
        if inputs.I_request > 0 % Discharge
            I_cmd = inputs.I_request * thermal_derate * min(soc_min_derate);
        else % Charge
            I_cmd = inputs.I_request * thermal_derate * min(soc_max_derate);
        end
    end

    %% 4. CONTROL LOGIC (Balancing & State Machine)
    V_avg = mean(inputs.V_cells);
    states.balancing_active = (inputs.V_cells > V_avg + 0.01) & (abs(inputs.I_measured) < 0.1);

    if fault_latched
        bms_state = 'Fault';
    elseif strcmp(inputs.Mode, 'Run')
        bms_state = 'Run';
    else
        bms_state = 'Standby';
    end

    states.bms_state = bms_state;
    states.I_cmd = I_cmd;
end
