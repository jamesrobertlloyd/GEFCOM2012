%%%% Temperature predictions based on annual average and recent trends
%%%% This version splits up the trend and periodic component for simplicity

%% Import data

data = importdata('time_series_raw.csv',',');
times = data(:,1);
temps = data(:,2:end);
clear data

%% Split data into testing and training

train = 1:39414;
test  = 39415:39600;
train = train';
test = test';

%% Remove means

means = mean(temps(train,:));
temps = temps - repmat(means, size(temps, 1), 1);

%% Scale the data

sds = std(temps(train,:));
temps = temps ./ repmat(sds, size(temps, 1), 1);

%% Predict based on past + GP (independent models)

for i = 1:11
  % How far to look back in time? Obtained by counting rows in spreadsheet
  one_year = 8784;
  two_years = 17544;
  three_years = 26304;
  four_years = 35064;
  % Construct the testing data
  window = 1000;
  train_indices = train((end-window+1):end);
  all_indices = [train_indices; test];
  train_times = times(train_indices);
  all_times = times(all_indices);
  train_temps = temps(train_indices, i);
  historical_temps = (temps(all_indices - one_year, i) + ...
                     temps(all_indices - two_years, i) + ...
                     temps(all_indices - three_years, i) + ...
                     temps(all_indices - four_years, i)) / 4;
  % Smooth the data - local linear regression with bandwidth of one day
  smoother = ksrlin (train_times, train_temps, 1, length(train_times));
  train_smooth = smoother.f';
  smoother = ksrlin (all_times, historical_temps, 1, length(all_times));
  historical_smooth = smoother.f';
  % Predict the smooth bit
  x = train_times;
  x_test = all_times;
  y = train_smooth - historical_smooth(1:length(train_times));
  % Setup the GP, train and predict
  covfunc = {@covSEiso};
  hyp.cov = [log(03) ; log(0.2)];
  likfunc = @likGauss;
  hyp.lik = log(0.05);
  meanfunc = @meanZero;
  hyp.mean = [];
%   hyp_opt = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
  hyp_opt = hyp;
                           
  exp(hyp_opt.cov)
  exp(hyp_opt.lik)
          
  [smooth_difference, ~] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, x, y, x_test);
  % Now predict the periodic bit
  % Construct the testing data
  window = 250; % More recent than the trend - want to capture current variation
  train_indices = train((end-window+1):end);
  all_indices = [train_indices; test];
  train_times = times(train_indices);
  all_times = times(all_indices);
  train_temps = temps(train_indices, i);
  all_temps = temps(all_indices, i);
  % Setup GP data
  x = train_times;
  x_test = all_times;
  y = train_temps - train_smooth((end-window+1):end);
  % Setup the GP, train and predict
  covfunc = {@covPeriodic};
  hyp.cov = [log(0.1) ; log(1) ; log(0.5)];
  likfunc = @likGauss;
  hyp.lik = log(0.15);
  meanfunc = @meanZero;
  hyp.mean = [];
  hyp_opt = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
                           
  exp(hyp_opt.cov)
  exp(hyp_opt.lik)
          
  [m, ~] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, x, y, x_test);
  prediction = m + historical_smooth((end-window-length(test)+1):end) + ...
                   smooth_difference((end-window-length(test)+1):end);
  % Plot
  
  h = figure;
  plot (train_times, train_temps, 'k');
  hold on
  plot (x_test, historical_smooth((end-window-length(test)+1):end), 'c');
  plot (x_test, historical_smooth((end-window-length(test)+1):end) + ...
                smooth_difference((end-window-length(test)+1):end), 'g');
  plot (x_test, prediction, 'r');
  xlim ([max(train_times) - 10, max(times)]);
  xlabel('Time')
  ylabel('Temperature')
  hold off
  getframe;
  %save2pdf ('temp_pred.pdf', h, 600);
  
  % Save prediction
  temps(test, i) = prediction((end-length(test)+1):end);
  
  % Wait a bit
  pause(1);
end

%% Compute smoothed temperatures

smoothed = zeros(size(temps));

for i = 1:11
  smooth = ksrlin (times, temps(:, i), 1, length(times));
  smoothed(:, i) = smooth.f;
end

save ('smooth_temp_GP.mat', 'smoothed');
csvwrite ('smooth_temp_GP.csv', smoothed);

%% Rescale the data

temps = temps .* repmat(sds, size(temps, 1), 1);

%% Add means back

temps = temps + repmat(means, size(temps, 1), 1);

%% Write to file

csvwrite ('GP_pred_temp.csv', temps);
