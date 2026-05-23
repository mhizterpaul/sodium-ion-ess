%% NFPP Physical Plant Builder (Simscape Equivalent)
% Ref: docs/paper.md
% Updates: 8-Tube Dual-Pump Loop, Horizontal Fin Rejection

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

        % Thermal: Sinusoidal 8-Tube Manifold
        plant.packs{p}.thermal.tubing.type = 'coolant_tubing.ssc';
        plant.packs{p}.thermal.tubing.count = 8;
    end

    % 3. Fluid & Rejection Loop
    % Loop: Pump1 -> 8 Tubes -> Pump2 -> Split -> 8 Tubes -> Hose -> Pump1
    plant.cooling.pump_system = 'Dual distributed BLDC (Pump1, Pump2)';
    plant.cooling.piping = '8-Microtube Al 3003 Sinusoidal Loop';
    plant.cooling.rejection.type = 'rejection_stage.ssc';
    plant.cooling.rejection.fins = 'Horizontal Aluminum';
    plant.cooling.rejection.bridge = 'Conducting plate (140mm wide)';

    % 4. ESS Dimensions (450x180x140 mm)
    plant.enclosure.type = 'aluminum_heat_sink.ssc';
    plant.enclosure.dims = [450, 180, 140];

    disp('Full ESS Digital Twin Built:');
    disp('  Topology: 16S1P (4 Packs of 4) with 8-Tube Dual-Pump Loop');
    disp('  Rejection: Horizontal Fins with Thermal Bridge to Inlets');
    disp('  Enclosure: Aluminum (450x180x140 mm)');
end
