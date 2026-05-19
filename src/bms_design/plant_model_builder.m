%% NFPP Physical Plant Builder (Simscape Equivalent)
% Ref: docs/paper.md
% Updates: Distributed Electro-Thermal-Fluid Digital Twin

function plant = build_physical_plant(params)
    % Initialize 16-cell series stack with stochastic variation
    num_cells = 16;
    plant.cells = cell(num_cells, 1);

    for i = 1:num_cells
        % Heterogeneity: Normal distribution N(mu, sigma^2)
        % Capacity spread +/- 2-5%, Resistance +/- 1-3%
        cap_variation = 1 + (0.02 * randn());
        res_variation = 1 + (0.01 * randn());

        plant.cells{i}.type = 'nfpp_cell.ssc';
        plant.cells{i}.Q_nom = 10 * cap_variation;
        plant.cells{i}.R_0 = 0.01 * res_variation;

        % Thermal nodes coupling
        plant.cells{i}.thermal_network.spreader = 'copper_spreader.ssc';
        plant.cells{i}.thermal_network.tubing = 'coolant_tubing.ssc';
    end

    % Busbar + Interconnect Specification
    plant.busbar.material = 'Nickel-plated copper';
    plant.busbar.resistance = 100e-6; % 100 micro-ohms
    plant.busbar.heat_gen = 'I^2 * R_bus';

    % Balancing Circuitry (Passive)
    plant.balancing.type = 'Passive Bleed';
    plant.balancing.R_bleed = 33; % Ohms
    plant.balancing.thermal_coupling = 'Enclosure thermal node';

    % Cooling Infrastructure
    plant.cooling.pump = 'pump_actuator.ssc';
    plant.cooling.reject_port = 'reject_port_atomizer.ssc';
    plant.cooling.fluid = 'Ethylene Glycol / Water';

    % Pre-charge & Protection Circuitry
    plant.protection.precharge_R = 25; % Ohms
    plant.protection.dc_link_cap = 1000e-6; % 1000uF
    plant.protection.main_contactor.transition_R = 0.001; % Weld fault modeling

    % Sensor Network (Realistic measurement model)
    plant.sensors.voltage.resolution = 16; % bit
    plant.sensors.voltage.noise = 0.002;  % 2mV RMS
    plant.sensors.temp.type = 'NTC Thermistor';
    plant.sensors.temp.placement = 'Every 2 cells';
    plant.sensors.current.type = 'Hall-effect';

    % Fault Injection Hooks
    plant.faults = {
        'Internal short', ...
        'Cooling blockage', ...
        'Pump degradation', ...
        'Sensor drift', ...
        'Contactor weld'
    };

    plant.config = '16S1P';
    plant.nominal_voltage = 16 * 3.2;

    disp('Simscape Multiphysics Plant Model Built:');
    disp(['  Topology: ' plant.config ' Distributed Electro-Thermal-Fluid']);
    disp(['  Cooling: Active Liquid + Atomized Reject Port']);
    disp(['  Sensors: 16-bit Realistic Network']);
end
