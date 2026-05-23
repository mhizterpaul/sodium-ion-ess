%% NFPP Physical Plant Builder (Standalone Cell Focus)
% Ref: docs/paper.md
% Updates: Standalone 16S1P pack without external cooling hardware.

function plant = build_physical_plant(params)
    % 1. Grid & PCCS Subsystem
    plant.pccs.type = 'Power Conversion and Conditioning System';

    % 2. Modular Pack Assembly (4 Packs of 4)
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
    plant.enclosure.type = 'Standalone NFPP Pack';
    plant.enclosure.dims = [450, 180, 140];

    disp('Full ESS Digital Twin Built:');
    disp('  Topology: 16S1P (4 Packs of 4)');
    disp('  Cooling: Natural/Forced Convection (Hardware-less)');
    disp('  Enclosure: Aluminum (450x180x140 mm)');
end
