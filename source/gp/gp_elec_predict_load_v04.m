%% GP prediction using 'electric' covariance function

%% Import load and temperature data

data = importdata('Z_t_with_zeros.csv',',');
times = data(:,1);
loads = data(:,2:end);
data = importdata('GP_pred_temp_008.csv',',');
temps = data(:,1:end);
data = load('smooth_temp_GP_08.mat', 'smoothed');
smooth_temps = data.smoothed;
clear data

%% Split into test and train

train = (loads(:,1) ~= 0);
test  = (loads(:,1) == 0);

%% Remove means

means = mean(loads(train,:));
loads = loads - repmat(means, size(loads, 1), 1);

means_t = mean(temps(train,:));
temps = temps - repmat(means_t, size(temps, 1), 1);

means_st = mean(smooth_temps(train,:));
smooth_temps = smooth_temps - repmat(means_st, size(smooth_temps, 1), 1);

%% Scale the data

sds = std(loads(train,:));
loads = loads ./ repmat(sds, size(loads, 1), 1);

sds_t = std(temps(train,:));
temps = temps ./ repmat(sds_t, size(temps, 1), 1);

sds_st = std(smooth_temps(train,:));
smooth_temps = smooth_temps ./ repmat(sds_st, size(smooth_temps, 1), 1);

%% Remove the 'errors' in zone 4

threshold = -2.7; % Found by querying data
error_indices = find((loads(:,4) < threshold) .* train);
loads(error_indices,4) = loads(error_indices-24, 4);

%% Main prediction code

for i = 1:size(loads, 2) % for each zone
  % Find the prediction regions in order
  searching_for_start = true;
  for j = 1:length(times)
    if searching_for_start
      if test(j)
        start_index = j-1;
        k = j + 1;
        searching_for_start = false;
        searching_for_end = true;
        while searching_for_end
          if (k > length(times)) || train(k) 
            searching_for_end = false;
            end_index = k;
          else
            k = k+1;
          end
        end
        % Found a region - try different models and combine to form
        % prediction
        lmls = zeros(size(temps, 2), 1);
        for t_i = 1:size(temps, 2)
          % Setup training and testing data splits
          if k < length(times)
            forecast = false;
            window = 500;
            train_times = [times((start_index - window):(start_index)) ; ...
                           times((end_index):(end_index + window))];
            train_loads = [loads((start_index - window):(start_index), i) ; ...
                           loads((end_index):(end_index + window), i)];
            train_temps = [temps((start_index - window):(start_index), t_i) ; ...
                           temps((end_index):(end_index + window), t_i)];
            train_smooth_temps = [smooth_temps((start_index - window):(start_index), t_i) ; ...
                           smooth_temps((end_index):(end_index + window), t_i)];
            test_times  = times((start_index+1):(end_index-1));
            test_temps  = temps((start_index+1):(end_index-1), t_i);
            test_smooth_temps  = smooth_temps((start_index+1):(end_index-1), t_i);
            test_indices = (start_index+1):(end_index-1);
            all_times = [times((start_index - window):(end_index + window))];
            all_temps = [temps((start_index - window):(end_index + window),:)];
          else
            forecast = true;
            window = 500;
            train_times = times((start_index - window):(start_index));
            train_loads = loads((start_index - window):(start_index), i);
            train_temps = temps((start_index - window):(start_index), t_i);
            train_smooth_temps = smooth_temps((start_index - window):(start_index), t_i);
            test_times  = times((start_index+1):(end_index-1));
            test_temps  = temps((start_index+1):(end_index-1), t_i);
            test_smooth_temps  = smooth_temps((start_index+1):(end_index-1), t_i);
            test_indices = (start_index+1):(end_index-1);
            all_times = [times((start_index - window):(end_index-1))];
            all_temps = [temps((start_index - window):(end_index-1), :)];
          end
          % Now do GP regression
          x = [train_times train_smooth_temps train_temps];
          x_test = [test_times test_smooth_temps test_temps];
          y = train_loads;
          
          if forecast
            if i == 9
              % This time series looks mostly random - use a simpler kernel
              x = x(:,1);
              x_test = x_test(:,1);
              covfunc = {@covSum, {@covSEiso, @covSEiso, @covPeriodic}};
              hyp.cov = [log(3) ; log(0.3) ; 
                         log(10) ; log(0.5) ; 
                         log(0.1) ; log(1) ; log(1)];
            else
              covfunc = @covElec01;
              % Equivalently - this kernel can be written in GPML
              % {@covSum, {{@covMask, {[1, 0, 0], @covSEiso}}, {@covMask, {[0, 1, 0], @covSEiso}},
              %            {@covProd, {{@covMask, {[0, 0, 1], @covSEiso}},
              %            {@covMask, {[1, 0, 0], @covPeriodic}}}}}
              hyp.cov = [log(2) ; log(0.3) ; 
                         log(0.5) ; log(0.5) ;
                         log(0.5) ; log(0.5) ;
                         log(0.1) ; log(1) ; log(1)];
            end
          else
            if i == 9
              % This time series looks mostly random - use a simpler kernel
              x = x(:,1);
              x_test = x_test(:,1);
              covfunc = {@covSum, {@covSEiso, @covSEiso, @covPeriodic}};
              hyp.cov = [log(3) ; log(0.3) ; 
                         log(10) ; log(0.5) ; 
                         log(0.1) ; log(1) ; log(1)];
            else
              covfunc = @covElec02;
              % Equivalently - this kernel can be written in GPML
              % {@covSum, {{@covMask, {[1, 0, 0], @covSEiso}}, {@covMask, {[0, 1, 0], @covSEiso}},
              %            {@covProd, {{@covMask, {[0, 0, 1], @covSEiso}},
              %                        {{@covMask, {[1, 0, 0], @covSEiso}}
              %                        {@covMask, {[1, 0, 0], @covPeriodic}}}}}
              hyp.cov = [log(2) ; log(0.3) ; 
                         log(0.5) ; log(0.5) ;
                         log(0.5) ; log(0.5) ;
                         log(2) ; log(0.5) ; 
                         log(0.1) ; log(1) ; log(1)];
            end
          end
          likfunc = @likGauss;
          if i == 9
            hyp.lik = log(0.8);
          else
            hyp.lik = log(0.2);
          end
          meanfunc = @meanZero;
          hyp.mean = [];
          hyp_opt = minimize(hyp, @gp, -1, @infExact, meanfunc, covfunc, likfunc, ...
                             x, y);
                           
          [m, ~] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, x, y, [x_test ; x]);

          if t_i == 1
            predictions = zeros(length(m), size(temps, 2));
          end
          predictions(:, t_i) = m(:);
          lmls(t_i) = -gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, x, y);

        end
        % Average predictions, plot and save
        weights = exp(lmls - max(lmls)) / sum(exp(lmls - max(lmls)));
        av_prediction = predictions * weights;
        
        % Plot
        
        best_i = find(weights == max(weights));
        h = figure;
        plot (train_times(1:(window+1)), train_loads(1:(window+1)), 'k');
%         [AX,H1,H2] = plotyy(train_times, train_loads, all_times, all_temps(:,best_i), 'plot');
        hold on
%         plot (train_times, av_prediction((length(test_times)+1):end), 'g');
        if ~forecast
          plot (train_times((window+2):end), train_loads((window+2):end), 'k');
        end
        plot (test_times, av_prediction(1:length(test_times)), 'r');
        xlim ([times(start_index+1)-5, times(end_index-1)+5]);
        xlabel('Time')
        ylabel('Load')
        hold off
%         set(get(AX(1),'xlim'),[times(start_index+1)-5, times(end_index-1)+5]);
%         set(get(AX(2),'xlim'),[times(start_index+1)-5, times(end_index-1)+5]);
%         set(get(AX(1),'Ylabel'),'String','Load') 
%         set(get(AX(2),'Ylabel'),'String','Temperature') 
        getframe;  
%         save2pdf ('load_pred.pdf', h, 600);
        h = figure;
        plot (all_times, all_temps(:,best_i), 'k');
        xlim ([times(start_index+1)-5, times(end_index-1)+5]);
        xlabel('Time')
        ylabel('Temperature')
        getframe;  
%         save2pdf ('best_temp.pdf', h, 600);

        % Save

        loads(test_indices, i) = av_prediction(1:length(test_indices));
        
        % Wait so we can see what's happening
        pause (1);
      end
    elseif train(j)
      searching_for_start = true;
    end
  end
end

%% Rescale the data

loads = loads .* repmat(sds, size(loads, 1), 1);

%% Add means back

loads = loads + repmat(means, size(loads, 1), 1);

%% Write to file

csvwrite ('GP_pred_covElec07.csv', loads);