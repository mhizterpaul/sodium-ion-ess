%% Network Simulator
% Ref: docs/paper.md section 3
% Attaches different topologies and load profiles to 2 feeders with phase measurements.

function results = network_simulator(topologies, load_profiles)
    % 1. Initialization
    if nargin < 1
        topologies = {'Radial', 'Ring'};
    end
    if nargin < 2
        load_profiles = {ones(24,1)*50, ones(24,1)*30}; % kW for Feeder 1 & 2
    end

    results = struct();
    results.timestamp = datestr(now);
    results.feeders = cell(2, 1);

    % 2. Topology and Load Attachment Loop
    for f = 1:2
        feeder_id = ['Feeder_', num2str(f)];
        topology = topologies{mod(f-1, length(topologies)) + 1};
        profile = load_profiles{f};

        % 3. Phase Change Measurements (Simulated)
        % V(t) = Vm * sin(wt + phi)
        % I(t) = Im * sin(wt + phi - theta) where cos(theta) is Power Factor
        pf = 0.95 - (rand() * 0.1); % 0.85 to 0.95
        phase_shift_deg = acosd(pf);

        voltage_phase = [0, -120, 120]; % Balanced 3-Phase
        current_phase = voltage_phase - phase_shift_deg;

        % 4. Result Aggregation
        results.feeders{f}.id = feeder_id;
        results.feeders{f}.topology = topology;
        results.feeders{f}.load_profile_kw = profile;
        results.feeders{f}.measurements.power_factor = pf;
        results.feeders{f}.measurements.phase_shift_deg = phase_shift_deg;
        results.feeders{f}.measurements.voltage_phases = voltage_phase;
        results.feeders{f}.measurements.current_phases = current_phase;
    end

    % 5. Export to JSON
    json_str = jsonencode(results);
    fid = fopen('network_results.json', 'w');
    if fid ~= -1
        fprintf(fid, '%s', json_str);
        fclose(fid);
        disp('Network simulation results exported to network_results.json');
    else
        warning('Could not open network_results.json for writing.');
    end
end
