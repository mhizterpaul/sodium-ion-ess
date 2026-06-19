%% MODEL-INFORMED ENERGY DISPATCH LAYER (Core Research Contribution)
% Focus: Real-time partitioning of stochastic solar power into physically constrained sinks.
% This module implements the Energy Decomposition and Dispatch Policy.

function [P_targets, states] = dispatch_controller(inputs, params)
    % inputs: struct {P_solar, P_load_req, SOC, SOH, T_bat, V_grid, f_grid, price, Copex, P_array}
    % params: struct {P_max_bat, P_max_dump, SOC_min, SOC_max, T_crit, eta_inv}

    persistent dispatch_state;
    persistent total_energy_delivered;

    if isempty(dispatch_state)
        dispatch_state = 'Normal';
        total_energy_delivered = 0;
    end

    %% 1. FUNDAMENTAL ENERGY DECOMPOSITION (Core Object)
    % P_solar(t) = P_load(t) + P_bat(t) + P_reactive(t) + P_harmonic(t) + P_dump(t) + P_loss(t)

    P_total_in = inputs.P_solar + inputs.P_array;

    % 1.6 Unavoidable Physical Inefficiency (P_loss)
    P_loss = P_total_in * (1 - params.eta_inv);
    P_available = P_total_in - P_loss;

    %% 2. CONSTRAINT EVALUATION (Stability Manifold)
    % 1.2 Electrochemical Buffering (P_bat) constraints
    % Limited by SOC, SOH, and thermal state
    thermal_limit = max(0, 1 - exp((inputs.T_bat - params.T_crit)/5));
    if inputs.SOC > params.SOC_max
        P_bat_max_charge = 0;
    else
        P_bat_max_charge = params.P_max_bat * thermal_limit * inputs.SOH;
    end

    if inputs.SOC < params.SOC_min
        P_bat_max_discharge = 0;
    else
        P_bat_max_discharge = params.P_max_bat * thermal_limit * inputs.SOH;
    end

    %% 3. DISPATCH POLICY (Partitioning Logic)
    % Objective: Maximize Plant Utilization U(t) subject to sustainability and stability

    % 4. Minimum Sustainable Throughput (MST)
    % MST = Copex / price
    if isfield(inputs, 'price') && inputs.price > 0
        MST = inputs.Copex / inputs.price;
    else
        % Default fallback if pricing not provided
        MST = inputs.P_load_req * 0.5;
    end

    % 1.3 Grid-Forming Stability Energy (P_reactive)
    % Not energy consumption, but field support.
    % Simplified proxy: P_reactive demand increases with V_grid/f_grid deviation
    P_reactive = abs(inputs.V_grid - 1.0) * 10.0 + abs(inputs.f_grid - 60.0) * 5.0;

    % 1.4 Unwanted Spectral Energy (P_harmonic)
    % Minimized penalty state
    P_harmonic = 0.01 * P_total_in; % Proportional to switching intensity

    P_remaining = P_available - P_reactive - P_harmonic;

    % Allocation Logic (Satisfying U >= MST)
    if P_remaining >= inputs.P_load_req
        P_load = inputs.P_load_req;
        P_surplus = P_remaining - P_load;

        % Charge battery with surplus
        P_bat = min(P_surplus, P_bat_max_charge);
        P_surplus = P_surplus - P_bat;

        % 1.5 Safety Dissipation Sink (P_dump)
        P_dump = min(P_surplus, params.P_max_dump);
    else
        % Deficit
        P_load = P_remaining;
        P_deficit = inputs.P_load_req - P_load;

        P_bat_support = min(P_deficit, P_bat_max_discharge);
        P_load = P_load + P_bat_support;
        P_bat = -P_bat_support;
        P_dump = 0;
    end

    % Calculate Total Utilization: U = P_load + |P_bat| + P_dump_equivalent
    % Note: P_bat usage counts for throughput whether charging or discharging
    U = P_load + abs(P_bat) + P_dump;

    %% 4. OUTPUTS & METRICS
    P_targets.P_load = P_load;
    P_targets.P_bat = P_bat;
    P_targets.P_reactive = P_reactive;
    P_targets.P_dump = P_dump;
    P_targets.P_loss = P_loss;
    P_targets.P_harmonic = P_harmonic;

    % Efficiency calculation
    total_energy_delivered = total_energy_delivered + P_load;
    states.stability_index = 1.0 - (P_reactive / (params.P_max_bat + 1e-6));

    % Sustainability & Economic Viability Check
    % Violation if U < MST
    states.MST = MST;
    states.utilization = U;
    states.efficiency = P_load / (P_total_in + 1e-6);
    states.viability_margin = (U - MST) / (MST + 1e-6);
    states.economic_status = U >= MST;
end
