%% Physical Power Plant Builder
% Ref: docs/paper.md
% Updates: Standalone 16S1P pack and integrated power conversion digital twin.

function plant = build_physical_plant(params)
    % 1. Grid & PCCS Subsystem
    plant.pccs.type = 'Power Conversion and Conditioning System';
    plant.pccs.components = {'STS', 'PQC', 'Isolated DC/DC', 'Solar Array'};

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
    plant.enclosure.type = 'Standalone NFPP ESS';
    plant.enclosure.dims = [450, 180, 140]; % mm

    disp('Full Hybrid Solar-Storage Power Plant Digital Twin Built:');
    disp('  Topology: 16S1P (4 Packs of 4) + Solar Hybrid');
    disp('  Enclosure: Poly-material (450x180x140 mm)');
    disp('  Hardware: STS, PQC, Buck-Boost DC/DC ready.');
end
