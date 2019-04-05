clear;
% sampling from a Gaussian process prior

  var = 1.; % variance (sigma^2)
  len = 15; % lengthscale (smoothness)
  dt = 1; % time-step size
  kernel = 'exp'; % 'exp', 'matern32' or 'se' 
  T = 200;
  t = linspace(1,T,T)';
  seed = 1;
  red = [0.7 0.3 0.4];blue = [0.4 0.3 0.7];green = [0.4 0.7 0.3];
  
  
  
  
%% Kernel-based sampling %%
tic; disp('kernel-based sampling');
  rng(seed);
  cov = zeros(T, T);
  for i=1:T
      for j=1:T
          r = abs(t(i) - t(j));
          if strcmp(kernel,'exp')
            cov(i, j) = var * exp(-r / len);
          elseif strcmp(kernel,'matern32')
            cov(i, j) = var * (1 + sqrt(3)*r/len) * exp((-sqrt(3)) * r / len);
          elseif strcmp(kernel,'se')
            cov(i, j) = var * exp((-1/2) * r^2 / len^2);
          end
      end
  end
  x_k = mvnrnd(zeros(1,size(cov,1)),cov);
toc  
  figure(1)
  subplot(2,1,1);cla
  plot(x_k,'Color',blue)
  title('Kernel-based sample'); legend off;
  
  
  


%% State space sampling %%
tic; disp('state-space sampling');
  rng(seed);
  cf_to_ss = str2func(strcat('cf_',kernel,'_to_ss'));
  [F,L,Qc,H,Pinf] = cf_to_ss(var, len); % calculate state-space model
  
  [A,Q] = lti_disc(F,L,Qc,dt); % discretise the model
  
  z = zeros(length(L),T);x_s = zeros(T,1);
  z(:,t(1)) = chol(Pinf)' * randn(length(L),1);
  x_s(t(1)) = H * z(:,t(1));
  for k=2:T
    z(:,t(k)) = A * z(:,t(k-1)) + chol(Q)' * randn(length(L),1);
    x_s(t(k)) = H * z(:,t(k));
  end
toc
  figure(1)
  subplot(2,1,2);cla();
  plot(x_s,'Color',green)
  title('State space sample')

  
  
  
  
  
  
  
  
  
  
  
  
  

%% functions
  
% calculating state space form of Gaussian process covariance functions
  % exponential
  function [F,L,Qc,H,Pinf] = cf_exp_to_ss(magnSigma2, lengthScale)
    F = -1/lengthScale;% Feedback matrix  
    L = 1;% Noise effect matrix
    Qc = 2*magnSigma2/lengthScale;% Spectral density
    H  = 1;% Observation model
    Pinf = magnSigma2;%Stationary covariance
  end
  
  function [F,L,Qc,H,Pinf] = cf_matern32_to_ss(magnSigma2, lengthScale)
    lambda = sqrt(3)/lengthScale;% Derived constants
    F = [0,          1;
         -lambda^2,  -2*lambda];% Feedback matrix
    L = [0;   1];% Noise effect matrix
    Qc = 12*sqrt(3)/lengthScale^3*magnSigma2;% Spectral density
    H = [1,   0];% Observation model
    Pinf = [magnSigma2, 0;
            0,          3*magnSigma2/lengthScale^2];% Stationary covariance
  end
  
  % squared exponential
  function [F,L,Qc,H,Pinf] = cf_se_to_ss(magnSigma2, lengthScale, N)
    if nargin < 3 || isempty(N), N = 6; end
    kappa = 1/2/lengthScale^2;% Derived constants
    fn = factorial(N);% Precalculate factorial
    Qc = magnSigma2*sqrt(pi/kappa)*fn*(4*kappa)^N;% Process noise spectral density
    p = zeros(1,2*N+1);% Make polynomial
    for n=0:N
      p(end - 2*n) = fn*(4*kappa)^(N-n)/factorial(n)/(-1)^(n);
    end
    r = roots(p);% All the coefficients of polynomial p are real so roots are of form a+/-ib
    a = poly(r(real(r) < 0));%which means they are symmetrically distributed in the complex plane.
    F = diag(ones(N-1,1),1);% Feedback matrix
    F(end,:) = -a(end:-1:2); % Controllable canonical form
    L = zeros(N,1); L(end) = 1;% Noise effect matrix
    H = zeros(1,N); H(1) = 1;% Observation model
    Pinf = lyap(F,L*Qc*L');%Stationary covariance
  end
  
  
  function [A,Q] = lti_disc(F,L,Q,dt)
    A = expm(F*dt);% Closed form integration of transition matrix
    n   = size(F,1);% Closed form integration of covariance by matrix fraction decomposition
    Phi = [F L*Q*L'; zeros(n,n) -F'];
    AB  = expm(Phi*dt)*[zeros(n,n);eye(n)];
    Q   = AB(1:n,:)/AB((n+1):(2*n),:);
  end