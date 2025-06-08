% Ensure MATPOWER is added to the path
addpath(genpath('matpower')); % Adjust the folder name if different

% Load MATPOWER and the IEEE 14-bus system case
mpc = loadcase('case14');

% Define the number of samples
num_samples = 100;

% Pre-allocate arrays for normal and compromised data
normal_data = zeros(num_samples, size(mpc.bus, 1) * size(mpc.bus, 2));
compromised_data_strong = zeros(num_samples, size(mpc.bus, 1) * size(mpc.bus, 2));

for i = 1:num_samples
    % Generate normal data
    mpc.bus(:, 3) = mpc.bus(:, 3) .* (1 + 0.1 * randn(size(mpc.bus(:, 3))));
    mpc = runpf(mpc);
    normal_data(i, :) = reshape(mpc.bus', 1, []);

    % Generate strong attack data
    attack_vector = create_attack_vector(mpc, 'strong');
    mpc = apply_attack(mpc, attack_vector);
    compromised_data_strong(i, :) = reshape(mpc.bus', 1, []);
end

% Create column labels
bus_attributes = {'BusNum', 'Type', 'Pd', 'Qd', 'Gs', 'Bs', 'AreaNum', 'Vm', 'Va', 'BaseKV', 'Zone', 'Vmax', 'Vmin'};
num_buses = size(mpc.bus, 1);
num_attributes = length(bus_attributes);
column_labels = cell(1, num_buses * num_attributes);
for i = 1:num_buses
    for j = 1:num_attributes
        column_labels{(i-1)*num_attributes + j} = sprintf('Bus%d_%s', i, bus_attributes{j});
    end
end

% Convert data to table and add column labels
normal_data_table = array2table(normal_data, 'VariableNames', column_labels);
compromised_data_strong_table = array2table(compromised_data_strong, 'VariableNames', column_labels);

% Save the generated data in CSV format with headers
writetable(normal_data_table, 'normal_data.csv');
writetable(compromised_data_strong_table, 'compromised_data_strong.csv');

% Helper functions
function attack_vector = create_attack_vector(mpc, attack_type)
    % Create an attack vector based on the attack type (strong)
    num_buses = size(mpc.bus, 1);
    attack_vector = zeros(num_buses, 1);
    
    switch attack_type
        case 'strong'
            attack_vector = 0.3 * randn(num_buses, 1); % Strong attack on bus data
    end
end

function mpc = apply_attack(mpc, attack_vector)
    % Apply the attack vector to the bus data
    mpc.bus(:, 3) = mpc.bus(:, 3) + attack_vector;
end
