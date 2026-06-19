%% Physical Power Plant Builder
% Ref: docs/paper.md
% Updates: Standalone 16S1P pack and integrated power conversion digital twin.

function plant = build_physical_plant(params)
    % 1. Grid & PCCS Subsystem
    plant.pccs.type = 'Power Conversion and Conditioning System';
    plant.pccs.components = {'STS', 'PQC', 'Isolated DC/DC'};

    % 2. Microgrid Generation Assets
    plant.generation.solar.model = 'Mono-crystalline PV';
    plant.generation.solar.capacity_kwp = 100;
    plant.generation.primary_array.capacity_kw = 50;

    % 3. Modular BESS Assembly (4 Packs of 4)
    num_packs = 4;
    plant.packs = cell(num_packs, 1);

    for p = 1:num_packs
        plant.packs{p}.id = ['Pack_' num2str(p)];
        plant.packs{p}.cells = cell(4, 1);
        for c = 1:4
            plant.packs{p}.cells{c}.type = 'nfpp_cell.ssc';
        end
    end

    % 3. Enclosure & Environment
    plant.enclosure.type = 'Standalone NFPP ESS';
    plant.enclosure.dims = [450, 180, 140]; % mm

    disp('Full Hybrid Solar-Storage Power Plant Digital Twin Built:');
    disp('  Generation: 100kWp Solar PV + 50kW Primary Array');
    disp('  BESS: 100kWh (16S1P Modular Stack)');
    disp('  Hardware: STS, PQC, Buck-Boost DC/DC ready.');
end
