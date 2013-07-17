%% Load data

Z = csvread('Z01_pred.csv');

%% Plot

h = figure;
set(h, 'Position', [0 0 800 400])
plot (1:33, Z(1:33), 'b-', 'LineWidth', 2);
hold on;
plot (34:201, Z(34:201), 'r--', 'LineWidth', 2);
legend('True load','GBM prediction')
plot (202:length(Z), Z(202:end), 'b-', 'LineWidth', 2);
plot ([33.5, 33.5], [min(Z), max(Z)], 'k--');
plot ([201.5, 201.5], [min(Z), max(Z)], 'k--');
xlim ([1, length(Z)]);
ylim ([min(Z), max(Z)]);
xlabel('Time');
ylabel('Load (Z01)');
set(gca, 'XTick', []);
getframe;
save2pdf ('Z01_pred.pdf', h, 600);