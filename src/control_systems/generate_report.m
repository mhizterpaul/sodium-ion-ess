function generate_report(results)
    fprintf('BMS Simulation Report\n');
    fprintf('=====================\n');
    fprintf('Cell Chemistry: NFPP Sodium-Ion\n');
    fprintf('Peak Temperature: %.2f C\n', results.max_temp);
    fprintf('Total Energy Processed: %.2f kWh\n', results.total_energy);
end
