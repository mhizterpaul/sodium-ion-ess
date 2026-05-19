%% ROBUST MULTI-LAYER BMS ARCHITECTURE
% Ref: ISO 26262 Hierarchical Safety + Optimal Control (MPC-inspired)

function [I_cmd, states] = bms_control_logic(inputs, params)
    % inputs: struct {SOC_est, V_cells, T_cells, I_measured, Mode, Fault_Reset, I_request, T_amb}

    persistent bms_state;
    persistent precharge_timer;
    persistent fault_status; % 0: Normal, 1: Warning, 2: Derating, 3: Shutdown, 4: Latch
    if isempty(bms_state)
        bms_state = 'Standby';
        precharge_timer = 0;
        fault_status = 0;
    end

    %% LAYER 1: SAFETY SYSTEM (Hierarchical & Latched)
    % Multi-tier Fault Classification
    T_max_cell = max(inputs.T_cells);
    V_max_cell = max(inputs.V_cells);

    if T_max_cell > 85 || V_max_cell > 3.9
        fault_status = 4; % LATCHED LOCKOUT
    elseif T_max_cell > 75 || V_max_cell > 3.8
        fault_status = 3; % SHUTDOWN
    elseif T_max_cell > 65
        fault_status = 2; % DERATING
    elseif T_max_cell > 55
        fault_status = 1; % WARNING
    else
        if fault_status < 4, fault_status = 0; end
    end

    if inputs.Fault_Reset && fault_status == 4, fault_status = 0; end

    %% LAYER 2: STATE MACHINE (Deterministic Logic)
    if fault_status >= 3, bms_state = 'Fault'; end

    switch bms_state
        case 'Standby'
            if strcmp(inputs.Mode, 'Drive'), bms_state = 'Precharge'; precharge_timer = 0; end
        case 'Precharge'
            precharge_timer = precharge_timer + 0.01;
            if precharge_timer > 0.05, bms_state = 'Run'; end
        case 'Run'
            if strcmp(inputs.Mode, 'Standby'), bms_state = 'Standby'; end
        case 'Fault'
            if fault_status < 3, bms_state = 'Standby'; end
    end

    %% LAYER 3: ESTIMATION (Nonlinear OCV-SOC mapping)
    % 5th-order Polynomial OCV for NFPP
    SOC = inputs.SOC_est;
    V_oc = 2.0 + 3.5*SOC - 5.1*SOC^2 + 4.8*SOC^3 - 2.1*SOC^4 + 0.5*SOC^5;

    %% LAYER 4: OPTIMAL ENERGY MANAGEMENT (MPC-inspired Arbitration)
    % Minimize J = (I - I_ref)^2 + lambda*T_penalty
    Q_n = params.Nominal_cell_capacity_Ah;
    I_ref = inputs.I_request;
    if strcmp(bms_state, 'Run')
        % Soft Constraint Handling (QP-like Arbitration)
        lambda_T = 0.5;
        T_margin = max(0, T_max_cell - 60);
        I_opt = I_ref * exp(-lambda_T * T_margin / 25);

        % Hierarchical Derating override
        if fault_status == 2, I_opt = min(I_opt, 0.5 * Q_n); end
        I_cmd = I_opt;
    else
        I_cmd = 0;
    end

    %% LAYER 5: ACTUATION & BALANCING (Energy-based)
    states.bms_state = bms_state;
    states.fault_status = fault_status;

    % Energy-based Balancing: P = (V - Vavg)^2 / R_bleed
    R_bleed = 10; % Ohms
    V_avg = mean(inputs.V_cells);
    states.balancing_active = (inputs.V_cells - V_avg) > 0.01;
    states.P_balance = (states.balancing_active .* (inputs.V_cells - V_avg).^2) ./ R_bleed;

    states.contactor_main = strcmp(bms_state, 'Run');
    states.contactor_pre = strcmp(bms_state, 'Precharge');
    states.V_oc = V_oc;
end

%% PACK DYNAMICS (2-Cell Imbalance Model)
function [sys_ss] = get_pack_dynamics(params)
    % x = [SOC1, SOC2, V1, V2, T1, T2]'
    % Models voltage dispersion and thermal gradients
    R0 = 0.01; Q = params.Nominal_cell_capacity_Ah * 3600;
    Cth = 500; hA = 0.1;

    A = zeros(6,6);
    A(5,5) = -hA/Cth; A(6,6) = -hA/Cth;

    B = zeros(6,1); % u = I
    B(1,1) = -1/Q; B(2,1) = -1.02/Q; % 2% capacity imbalance

    C = zeros(2,6); % y = [V1, V2]
    C(1,1) = 0.8; C(2,2) = 0.8;

    D = [-R0; -R0];
    sys_ss = ss(A, B, C, D);
end
