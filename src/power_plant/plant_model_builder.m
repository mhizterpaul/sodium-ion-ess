%% Physical Power Plant Builder
% Ref: docs/paper.md
% Updates: Multi-Feeder Shared Source Digital Twin with 2 Feeders

function plant = build_physical_plant(params)
    % 0. Import Data from Pipeline (Mandatory Dependency)
    script_dir = fileparts(mfilename('fullpath'));
    data_file = fullfile(script_dir, 'cell_params.json');

    if ~exist(data_file, 'file')
        error('Mandatory pipeline artifact missing: %s. Run simulation/envelope.py first.', data_file);
    end

    fid = fopen(data_file, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    data_decoded = jsondecode(str);

    if ~isfield(data_decoded, 'ssc_params')
        error('Invalid data structure in %s. Missing ssc_params.', data_file);
    end
    ssc_params = data_decoded.ssc_params;

    % 1. Shared Source Architecture (Solar + BESS)
    plant.sources.shared_bus.type = 'Balanced 3-Phase DC-Link AC Coupling';
    plant.sources.solar.capacity_kwp = 100;
    plant.sources.bess.capacity_kwh = 100;

    % 2. Shared Power Conditioning Unit (PCU)
    plant.pccs.pcu.type = 'central_inverter_pcu.ssc';
    plant.pccs.pcu.rating_kva = 150;
    plant.pccs.transformer.type = 'step_up_transformer.ssc';
    plant.pccs.switchgear.type = 'mv_switchgear.ssc';

    % 3. Multi-Feeder Distribution Network (2 Feeders)
    plant.network.feeders = cell(2, 1);
    for i = 1:2
        plant.network.feeders{i}.id = ['Feeder_', num2str(i)];
        plant.network.feeders{i}.model = 'feeder.ssc';
        plant.network.feeders{i}.monitoring = 'Phase Dynamics (XR)';
    end
    plant.network.topology = 'Radial Feeders from Shared PCC';

    % 4. Modular AC-Coupled BESS Assembly (208 Modules)
    num_modules = 208;
    plant.bess.modules = cell(num_modules, 1);
    for m = 1:num_modules
        plant.bess.modules{m}.id = ['Module_' num2str(m)];
        plant.bess.modules{m}.type = 'nfpp_cell.ssc';

        % Assign DFN-Informed Parameters from Pipeline
        plant.bess.modules{m}.R_0 = ssc_params.R_0;
        plant.bess.modules{m}.V_nom = ssc_params.V_nom;
        plant.bess.modules{m}.Q_nom = ssc_params.Q_nom;
        plant.bess.modules{m}.R1 = ssc_params.R1;
        plant.bess.modules{m}.C1 = ssc_params.C1;
        plant.bess.modules{m}.R2 = ssc_params.R2;
        plant.bess.modules{m}.C2 = ssc_params.C2;
        plant.bess.modules{m}.C_th_core = ssc_params.C_th_core;
    end

    % 5. Enclosure & Environment
    plant.enclosure.type = '20ft ISO Containerized Utility-Scale ESS';
    plant.enclosure.dims_mm = [6058, 2438, 2591];

    disp('--- Multi-Feeder Plant-Network Digital Twin Initialization ---');
    disp(['  Source: ', data_file]);
    disp('  Architecture: Shared Solar-BESS (2 coupled feeders: feeder.ssc)');
    disp(['  BESS: ', num2str(num_modules), ' Modular Units (100kWh Class)']);
    disp('  Diagnostic Target: Multi-Feeder State Realization (Phase Dynamics).');
end
