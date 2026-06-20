%% Physical Power Plant Builder
% Ref: docs/paper.md
% Updates: Standalone 16S1P pack and integrated power conversion digital twin.

function plant = build_physical_plant(params)
    % 0. Import Data from Pipeline (Mandatory Dependency)
    % Find file relative to script location
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

    % 1. Utility-Scale Interconnection & PCU
    plant.pccs.type = 'Utility-Scale Power Conditioning & Interconnection';
    plant.pccs.pcu.type = 'central_inverter_pcu.ssc';
    plant.pccs.pcu.rating_kva = 150;
    plant.pccs.transformer.type = 'step_up_transformer.ssc';
    plant.pccs.transformer.ratio = '415V/11kV';
    plant.pccs.switchgear.type = 'mv_switchgear.ssc';

    % 2. Microgrid Generation Assets
    plant.generation.solar.model = 'Mono-crystalline PV';
    plant.generation.solar.capacity_kwp = 100;
    plant.generation.primary_array.capacity_kw = 50;

    % 3. Representation Loads & Fault Injection
    plant.loads.type = 'Utility-Scale R-L Load';
    plant.loads.model = 'utility_load.ssc';
    plant.loads.p_nom_kw = 50;
    plant.loads.q_nom_kvar = 20;
    plant.faults.injection_hooks = {'Impedance Shift', 'Efficiency Drop', 'Sensor Drift'};

    % 4. Modular AC-Coupled BESS Assembly (208 Modules)
    num_modules = 208;
    plant.bess.modules = cell(num_modules, 1);
    plant.bess.coupling = 'AC-Coupled via BESS-PCU';

    for m = 1:num_modules
        plant.bess.modules{m}.id = ['Module_' num2str(m)];
        plant.bess.modules{m}.type = 'nfpp_cell.ssc';
        plant.bess.modules{m}.interface = 'central_inverter_pcu.ssc';

        % Assign Pipeline-Informed Parameters (Enforced)
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
    plant.enclosure.type = 'Containerized Utility-Scale ESS';
    plant.enclosure.dims = [6058, 2438, 2591]; % mm (20ft ISO)

    disp('Full Hybrid Solar-Storage Power Plant Digital Twin Built:');
    disp(['  Generation: 100kWp Solar PV + 50kW Primary Array (Data source: ', data_file, ')']);
    disp('  BESS: 100kWh (208 Modular 16S1P Units)');
    disp('  Loads: 50kW Resistive + 20kVAr Inductive (with Fault Injection)');
    disp('  Architecture: Utility-Scale (PCU, Step-up XFMR, MV Switchgear) ready.');
end
