%% NFPP Physical Plant Builder (Simscape Equivalent)
% Ref: docs/paper.md
% Updates: Al 3003 Dual-Tube, Poly-only Casing, 3-Airway Topology

function plant = build_physical_plant(params)
    % 1. Grid & Power Conditioning Layer
    plant.grid.type = 'grid_interface.ssc';
    plant.sts.type = 'static_transfer_switch.ssc';
    plant.pqc.type = 'power_quality_conditioner.ssc';

    % 2. Power Conversion Layer
    plant.pcs.type = 'bidirectional_dc_dc.ssc';

    % 3. Battery Pack & Thermal Layer
    num_cells = 16;
    plant.cells = cell(num_cells, 1);
    for i = 1:num_cells
        cap_variation = 1 + (0.02 * randn());
        res_variation = 1 + (0.01 * randn());

        plant.cells{i}.type = 'nfpp_cell.ssc';
        plant.cells{i}.casing = 'Poly-material (no Al laminate)';
        plant.cells{i}.Q_nom = 10 * cap_variation;
        plant.cells{i}.R_0 = 0.01 * res_variation;

        % Thermal coupling
        plant.cells{i}.thermal.spreader.type = 'copper_spreader.ssc';
        plant.cells{i}.thermal.tubing.type = 'coolant_tubing.ssc';
        plant.cells{i}.thermal.tubing.material = 'Aluminum Alloy 3003';
        plant.cells{i}.thermal.tubing.path = 'Sinusoidal (45% coverage)';
    end

    % 4. Active Cooling & Rejection Stage
    plant.cooling.pump = 'pump_actuator.ssc';
    plant.cooling.fluid = '60% Deionized Water / 40% Ethylene Glycol';
    plant.cooling.atomizers = '2 per side (4 total)';
    plant.cooling.topology = '3-Airway (2 Inlets at 1/3, 1 Exit at Back)';
    plant.cooling.airway_geometry = 'Oblong Rectangles (45% height coverage)';

    % 5. Interconnects & Sensors
    plant.busbar.resistance = 150e-6;
    plant.sensors.voltage.precision = '16-bit';

    % 6. Fault Injection Framework
    plant.faults = {
        'Voltage Sag', 'Frequency Instability', ...
        'Coolant Leak', 'Airway Blockage', ...
        'Internal short'
    };

    plant.config = '16S1P';
    plant.nominal_voltage = 16 * 3.2;

    disp('ESS Digital Twin Model Built:');
    disp('  Thermal: Al 3003 Dual-Tube Sinusoidal Network');
    disp('  Coolant: 60/40 Water-Glycol Mixture');
    disp('  Airflow: 3-Airway Draft Logic (45% height coverage)');
end
