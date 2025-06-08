% Add MATPOWER to MATLAB path (adjust the path as necessary)
addpath(genpath('matpower'));

% Initialize MATPOWER with IEEE 14-bus case
mpc = case14;

% Number of samples
num_samples = 1000;

% Initialize matrices to store all samples
all_bus_data_normal = [];
all_branch_data_normal = [];
all_bus_data_compromised = [];
all_branch_data_compromised = [];

% Compute admittance matrix Ybus
[Ybus, ~, ~] = makeYbus(mpc);

for sample = 1:num_samples
    % Generate normal data
    % Perturb bus data slightly with subtle random perturbations
    bus_data_normal = mpc.bus;
    bus_data_normal(:, 8) = bus_data_normal(:, 8) .* (1 + 0.001 * randn(size(bus_data_normal(:, 8)))); % Voltage magnitude (pu)
    bus_data_normal(:, 9) = bus_data_normal(:, 9) + 0.1 * randn(size(bus_data_normal(:, 9))); % Voltage angle (degrees)
    bus_data_normal(:, 9) = deg2rad(bus_data_normal(:, 9)); % Convert angle to radians

    % Perturb branch data slightly with subtle random perturbations
    branch_data_normal = mpc.branch;
    branch_data_normal(:, 3) = branch_data_normal(:, 3) .* (1 + 0.01 * randn(size(branch_data_normal(:, 3)))); % Resistance (pu)
    branch_data_normal(:, 4) = branch_data_normal(:, 4) .* (1 + 0.01 * randn(size(branch_data_normal(:, 4)))); % Reactance (pu)

    % Generate compromised data
    bus_data_compromised = bus_data_normal;
    % Apply stronger perturbations for compromised data
    bus_data_compromised(:, 8) = bus_data_compromised(:, 8) .* (1 + 0.01 * randn(size(bus_data_compromised(:, 8)))); % Voltage magnitude (pu)
    bus_data_compromised(:, 9) = bus_data_compromised(:, 9) + 1 * randn(size(bus_data_compromised(:, 9))); % Voltage angle (degrees)
    bus_data_compromised(:, 9) = deg2rad(bus_data_compromised(:, 9)); % Convert angle to radians

    branch_data_compromised = branch_data_normal;
    % Apply stronger perturbations for compromised data
    branch_data_compromised(:, 3) = branch_data_compromised(:, 3) .* (1 + 0.1 * randn(size(branch_data_compromised(:, 3)))); % Resistance (pu)
    branch_data_compromised(:, 4) = branch_data_compromised(:, 4) .* (1 + 0.1 * randn(size(branch_data_compromised(:, 4)))); % Reactance (pu)

    % Initialize Pi and Qi
    Pi_normal = zeros(size(bus_data_normal, 1), 1);
    Qi_normal = zeros(size(bus_data_normal, 1), 1);
    Pi_compromised = zeros(size(bus_data_compromised, 1), 1);
    Qi_compromised = zeros(size(bus_data_compromised, 1), 1);

    % Compute Pi and Qi based on power flow equations
    for i = 1:size(bus_data_normal, 1)
        Vi = bus_data_normal(i, 8);
        theta_i = bus_data_normal(i, 9);
        for j = 1:size(bus_data_normal, 1)
            if i ~= j
                Vj = bus_data_normal(j, 8);
                theta_j = bus_data_normal(j, 9);
                Gij = real(Ybus(i, j));
                Bij = imag(Ybus(i, j));
                theta_ij = theta_i - theta_j;
                Pi_normal(i) = Pi_normal(i) + Vi * Vj * (Gij * cos(theta_ij) + Bij * sin(theta_ij));
                Qi_normal(i) = Qi_normal(i) + Vi * Vj * (Gij * sin(theta_ij) - Bij * cos(theta_ij));
            end
        end
        Pi_normal(i) = Pi_normal(i) + Vi^2 * real(Ybus(i, i));
        Qi_normal(i) = Qi_normal(i) - Vi^2 * imag(Ybus(i, i));
    end

    for i = 1:size(bus_data_compromised, 1)
        Vi = bus_data_compromised(i, 8);
        theta_i = bus_data_compromised(i, 9);
        for j = 1:size(bus_data_compromised, 1)
            if i ~= j
                Vj = bus_data_compromised(j, 8);
                theta_j = bus_data_compromised(j, 9);
                Gij = real(Ybus(i, j));
                Bij = imag(Ybus(i, j));
                theta_ij = theta_i - theta_j;
                Pi_compromised(i) = Pi_compromised(i) + Vi * Vj * (Gij * cos(theta_ij) + Bij * sin(theta_ij));
                Qi_compromised(i) = Qi_compromised(i) + Vi * Vj * (Gij * sin(theta_ij) - Bij * cos(theta_ij));
            end
        end
        Pi_compromised(i) = Pi_compromised(i) + Vi^2 * real(Ybus(i, i));
        Qi_compromised(i) = Qi_compromised(i) - Vi^2 * imag(Ybus(i, i));
    end

    % Add sample number to bus and branch data
    sample_bus_data_normal = [repmat(sample, size(bus_data_normal, 1), 1), bus_data_normal(:, [1, 8, 9]), Pi_normal, Qi_normal];
    sample_branch_data_normal = [repmat(sample, size(branch_data_normal, 1), 1), branch_data_normal(:, [1, 2, 3, 4])];

    sample_bus_data_compromised = [repmat(sample, size(bus_data_compromised, 1), 1), bus_data_compromised(:, [1, 8, 9]), Pi_compromised, Qi_compromised];
    sample_branch_data_compromised = [repmat(sample, size(branch_data_compromised, 1), 1), branch_data_compromised(:, [1, 2, 3, 4])];

    % Append to all samples data
    all_bus_data_normal = [all_bus_data_normal; sample_bus_data_normal];
    all_branch_data_normal = [all_branch_data_normal; sample_branch_data_normal];

    all_bus_data_compromised = [all_bus_data_compromised; sample_bus_data_compromised];
    all_branch_data_compromised = [all_branch_data_compromised; sample_branch_data_compromised];
end

% Convert to tables
bus_table_normal = array2table(all_bus_data_normal, 'VariableNames', {'Sample', 'Bus', 'Voltage', 'Angle', 'Pi', 'Qi'});
branch_table_normal = array2table(all_branch_data_normal, 'VariableNames', {'Sample', 'From', 'To', 'Resistance', 'Reactance'});

bus_table_compromised = array2table(all_bus_data_compromised, 'VariableNames', {'Sample', 'Bus', 'Voltage', 'Angle', 'Pi', 'Qi'});
branch_table_compromised = array2table(all_branch_data_compromised, 'VariableNames', {'Sample', 'From', 'To', 'Resistance', 'Reactance'});

% Write to CSV files
writetable(bus_table_normal, 'all_buses_normal.csv');
writetable(branch_table_normal, 'all_branches_normal.csv');

writetable(bus_table_compromised, 'all_buses_compromised.csv');
writetable(branch_table_compromised, 'all_branches_compromised.csv');

% Display message
disp('Normal and compromised synthetic samples based on IEEE 14-bus system generated and saved to CSV files.');
