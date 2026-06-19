%% Physical Power Plant Builder
% Ref: docs/paper.md
% Updates: Standalone 16S1P pack and integrated power conversion digital twin.

function plant = build_physical_plant(params)
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

    % 3. Modular AC-Coupled BESS Assembly (208 Modules)
    num_modules = 208;
    plant.bess.modules = cell(num_modules, 1);
    plant.bess.coupling = 'AC-Coupled via BESS-PCU';

    for m = 1:num_modules
        plant.bess.modules{m}.id = ['Module_' num2str(m)];
        plant.bess.modules{m}.type = 'nfpp_cell.ssc';
        plant.bess.modules{m}.interface = 'central_inverter_pcu.ssc';
    end

    % 4. Enclosure & Environment
    plant.enclosure.type = 'Standalone NFPP ESS';
    plant.enclosure.dims = [450, 180, 140]; % mm

    disp('Full Hybrid Solar-Storage Power Plant Digital Twin Built:');
    disp('  Generation: 100kWp Solar PV + 50kW Primary Array');
    disp('  BESS: 100kWh (208 Modular 16S1P Units)');
    disp('  Architecture: Utility-Scale (PCU, Step-up XFMR, MV Switchgear) ready.');
end
