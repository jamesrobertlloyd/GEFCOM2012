function K = covElec01(hyp, x, z, i)

% Specialised covarianece for electric load forecasting
% Designed to work within GPML framework
% 
% James Lloyd, 2012

if nargin<2, K = '9'; return; end                  % report number of parameters
if nargin<3 || isempty(z)
  z1 = [];
  z2 = [];
  z3 = [];
else
  z1 = z(:,1);
  z2 = z(:,2);
  z3 = z(:,3);
end

x1 = x(:,1);
x2 = x(:,2);
x3 = x(:,3);

if nargin<4                                                        % covariances
  K = covSEiso (hyp(1:2), x1, z1) + ... % smooth in time
      covSEiso (hyp(3:4), x2, z2) + ... % smooth in smoothed temperature
      covSEiso (hyp(5:6), x3, z3) .* ... % temperature modulated...
      covPeriodic (hyp(7:9), x1, z1); % ...periodic in time
else                                                               % derivatives
  if i <= 2
    K = covSEiso (hyp(1:2), x1, z1, i);
  elseif i <= 4
    K = covSEiso (hyp(3:4), x2, z2, i-2);
  elseif i <= 6
    K = covSEiso (hyp(5:6), x3, z3, i-4) .* covPeriodic (hyp(7:9), x1, z1);
  elseif i <= 9
    K = covSEiso (hyp(5:6), x3, z3) .* covPeriodic (hyp(7:9), x1, z1, i-6);
  else
    error('Referring to out of range hyperparameter')
  end
end