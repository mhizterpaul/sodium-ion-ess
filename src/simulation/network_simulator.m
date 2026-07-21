%% Multi-Feeder Network Simulator (Refactored)
% Ref: docs/paper.md
% Attaches different topologies and varying RLC load profiles to 2 feeders.
% Shared Sources: Solar PV and BESS.
% Output: Phase change measurements and JSON result export.

function results = network_simulator(p_solar, p_bess)
    % 1. Initialization
    if nargin < 1, p_solar = 80; end % kW
    if nargin < 2, p_bess = 20; end  % kW

    num_feeders = 2;
    topologies = {'Radial', 'Ring'};
    duration_hours = 24;

    results = struct();
    results.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
    results.shared_source.p_solar_kw = p_solar;
    results.shared_source.p_bess_kw = p_bess;
    results.shared_source.p_total_kw = p_solar + p_bess;

    results.feeders = cell(num_feeders, 1);
    network_realization_state = zeros(num_feeders, 1);

    % 2. Multi-Feeder Loop
    total_p_load = 0;
    for f = 1:num_feeders
        feeder_id = f;
        topology = topologies{mod(f-1, length(topologies)) + 1};

        % 3. Varying RLC Load Profiles
        % P (Real Power), Q (Reactive Power), THD (Harmonics)
        p_profile = 30 + 20 * rand(duration_hours, 1);
        q_profile = 10 + 5 * randn(duration_hours, 1);
        thd_profile = 2 + rand(duration_hours, 1);

        total_p_load = total_p_load + mean(p_profile);

        % 4. Phase Change Measurements
        % theta = atan(Q/P)
        theta_deg = atand(q_profile ./ p_profile);

        % Relative Phase Deviation (delta theta) relative to nominal 0.95 PF
        nominal_theta = acosd(0.95);
        delta_theta = theta_deg - nominal_theta;

        % 5. Anomaly Detection (Phase-based)
        anomalies = abs(delta_theta) > 5.0;

        % Result Aggregation
        results.feeders{f}.feeder_id = feeder_id;
        results.feeders{f}.topology = topology;
        results.feeders{f}.p_kw = p_profile;
        results.feeders{f}.q_kvar = q_profile;
        results.feeders{f}.thd_percent = thd_profile;
        results.feeders{f}.theta_deg = theta_deg;
        results.feeders{f}.delta_theta = delta_theta;
        results.feeders{f}.anomalies = anomalies;

        network_realization_state(f) = mean(delta_theta);
    end

    results.network_realization_state_xr = network_realization_state;
    results.p_loss_est_kw = results.shared_source.p_total_kw - total_p_load;

    % 6. Export to JSON
    json_str = jsonencode(results);
    fid = fopen('network_realization.json', 'w');
    if fid ~= -1
        fprintf(fid, '%s', json_str);
        fclose(fid);
        disp('Multi-feeder network realization exported to network_realization.json');
    else
        warning('Could not open network_realization.json for writing.');
    end
end
