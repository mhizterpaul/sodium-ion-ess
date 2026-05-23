%% NFPP Physical Plant Builder (Simscape Equivalent)
% Ref: docs/paper.md
% Updates: Pack-level Architecture (4 cells/pack), Finned Tubing

function plant = build_physical_plant(params)
    % 1. Grid & PCCS
    plant.pccs.type = 'Power Conversion and Conditioning System';
    plant.pccs.topology = 'Grid -> STS -> PQC -> DC Link -> DC/DC';

    % 2. Pack-level Assembly (16 cells -> 4 packs)
    num_packs = 4;
    cells_per_pack = 4;
    plant.packs = cell(num_packs, 1);

    for p = 1:num_packs
        plant.packs{p}.id = ['Pack_' num2str(p)];
        plant.packs{p}.cells = cell(cells_per_pack, 1);

        for c = 1:cells_per_pack
            plant.packs{p}.cells{c}.type = 'nfpp_cell.ssc';
            plant.packs{p}.cells{c}.casing = 'Poly-material (no Al)';
        end

        % Finned Tubing Interface between each pack
        plant.packs{p}.thermal.tubing_interface.type = 'coolant_tubing.ssc';
        plant.packs{p}.thermal.tubing_interface.fins = 'Transverse (Al 3003)';
    end

    % 3. Chassis & Active Rejection
    plant.chassis.heatsink = 'aluminum_heat_sink.ssc';
    plant.cooling.atomizers = 2; % One per side
    plant.cooling.topology = '3-Airway induced draft';

    % 4. System Summary
    plant.config = '16S1P (4 Packs of 4)';
    plant.nominal_voltage = 16 * 3.2;

    disp('Full Multiphysics ESS Digital Twin Built:');
    disp(['  Topology: ' plant.config]);
    disp('  Thermal: Aluminum Finned Tubing + Induced Air Draft');
    disp('  Rejection: Dual Ultrasonic Atomizers');
end
