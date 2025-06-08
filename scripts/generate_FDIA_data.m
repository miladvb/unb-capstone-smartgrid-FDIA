% Ensure MATPOWER is added to the path
addpath(genpath('matpower')); % Adjust the folder name if different

% Load MATPOWER and the IEEE 14-bus system case
mpc = loadcase('case14');

% Define the number of samples
num_samples = 10000;

% Pre-allocate arrays for normal and compromised data
normal_data = zeros(num_samples, size(mpc.bus, 1) + size(mpc.branch, 1));
compromised_data_weak = zeros(num_samples, size(mpc.bus, 1) + size(mpc.branch, 1));
compromised_data_strong = zeros(num_samples, size(mpc.bus, 1) + size(mpc.branch, 1));

for i = 1:num_samples
    % Generate normal data
    mpc.bus(:, 3) = mpc.bus(:, 3) .* (1 + 0.1 * randn(size(mpc.bus(:, 3))));
    mpc = runpf(mpc);
    normal_data(i, :) = [mpc.bus(:, 8); mpc.branch(:, 14)];
    
    % Generate weak attack data
    attack_vector = create_attack_vector(mpc, 'weak');
    mpc = apply_attack(mpc, attack_vector);
    compromised_data_weak(i, :) = [mpc.bus(:, 8); mpc.branch(:, 14)];
    
    % Generate strong attack data
    attack_vector = create_attack_vector(mpc, 'strong');
    mpc = apply_attack(mpc, attack_vector);
    compromised_data_strong(i, :) = [mpc.bus(:, 8); mpc.branch(:, 14)];
end

% Save the generated data
save('normal_data.mat', 'normal_data');
save('compromised_data_weak.mat', 'compromised_data_weak');
save('compromised_data_strong.mat', 'compromised_data_strong');

% Helper functions
function attack_vector = create_attack_vector(mpc, attack_type)
    % Create an attack vector based on the attack type (weak, strong)
    num_buses = size(mpc.bus, 1);
    num_branches = size(mpc.branch, 1);
    
    attack_vector = zeros(num_buses + num_branches, 1);
    
    switch attack_type
        case 'weak'
            attack_vector(1:num_buses) = 0.1 * randn(num_buses, 1); % Weak attack on bus data
            attack_vector(num_buses+1:end) = 0.05 * randn(num_branches, 1); % Weak attack on branch data
        case 'strong'
            attack_vector(1:num_buses) = 0.3 * randn(num_buses, 1); % Strong attack on bus data
            attack_vector(num_buses+1:end) = 0.1 * randn(num_branches, 1); % Strong attack on branch data
    end
end

function mpc = apply_attack(mpc, attack_vector)
    % Apply the attack vector to the MATPOWER case
    mpc.bus(:, 3) = mpc.bus(:, 3) + attack_vector(1:size(mpc.bus, 1));
    mpc.branch(:, 14) = mpc.branch(:, 14) + attack_vector(size(mpc.bus, 1)+1:end);
end
