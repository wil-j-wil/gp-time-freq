clear;
% sampling from the time-frequency model prior
  
  % parameters of the 3 periodic components / filters
  var =   [0.1   0.2   0.4]; % variance (sigma^2)
  len =   [30    15    50]; % lengthscale (smoothness)
  omega = [pi/4  pi/8  pi/12]; % centre frequency
  
  dt = 1; % time-step size
  kernel = 'matern32'; % 'exp', 'matern32' or 'se' 
  T = 400;
  t = linspace(1,T,T)';
  red = [0.7 0.3 0.4];blue = [0.4 0.3 0.7];green = [0.4 0.7 0.3];
  
  
  
  
%% Kernel-based sampling %%
tic; disp('kernel-based sampling');
  cov = zeros(T, T);
  for i=1:T
      for j=1:T
          r = abs(t(i) - t(j));
          if strcmp(kernel,'exp')
            cov(i, j) = sum(var .* cos(omega*r) .* exp(-r ./ len));
          elseif strcmp(kernel,'matern32')
            cov(i, j) = sum(var .* cos(omega*r) .* (1 + sqrt(3)*r./len) .* exp((-sqrt(3)) * r ./ len));
          elseif strcmp(kernel,'se')
            cov(i, j) = sum(var .* cos(omega*r) .* exp((-1/2) * r^2 ./ len.^2));
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
  D = length(omega);
  [A,Q,H,Pinf,K,tau1] = get_disc_model(len,var,omega,D,kernel);
  
  z = zeros(length(H),T);x_s = zeros(T,1);
  z(:,t(1)) = chol(Pinf)' * randn(length(H),1);
  x_s(t(1)) = H * z(:,t(1));
  for k=2:T
    z(:,t(k)) = A * z(:,t(k-1)) + chol(Q)' * randn(length(H),1);
    x_s(t(k)) = H * z(:,t(k));
  end
toc
  figure(1)
  subplot(2,1,2);cla
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
  
  
  function [A,Q,H,Pinf,K,tau1] = get_disc_model(lenx,varx,omega,D,kernel)
  % Define the hyperparameters
  dt = 1;  % step size is 1 sample, regardless of sample rate
  w = omega;  % om
  
  lengthScale = lenx;
  magnSigma2 = varx;
  
  Qc1 = zeros(D,1);
  F1=[];L1=[];H1=[];Pinf1=[];
  for d=1:D
    % Construct the continuous-time SDE (from Solin+Sarkka AISTATS 2014)
    cf_to_ss = str2func(strcat('cf_',kernel,'_to_ss'));
    [F1d,L1d,Qc1d,H1d,Pinf1d] = cf_to_ss(magnSigma2(d), lengthScale(d));
    F1 = blkdiag(F1, F1d);
    L1 = vertcat(L1,L1d);
    Qc1(d) = Qc1d;
    H1 = horzcat(H1,H1d);
    Pinf1 = blkdiag(Pinf1,Pinf1d);
  end
  tau1 = length(L1d); % tau = model order (1 for Exponential, 2 for Matern 3/2, etc.)
  % Construct the continuous-time SDE: periodic (just a sinusoid) (from Solin+Sarkka AISTATS 2014)
  tau2=2; % real + imaginary
  F2=[];L2=[];Qc2=[];H2=[];
  for d=1:D
    F2 = blkdiag(F2,[0 -w(d); w(d) 0]);
    L2 = blkdiag(L2,eye(tau2));
    Qc2 = blkdiag(Qc2,zeros(tau2));
    H2 = horzcat(H2,[1 0]);
  end

  F2_kron=[];L=[];Qc=[];
  for d=1:D  % for higher-order models we must iterate to stack the kronecker products along the diagonal
    idx1 = tau1*(d-1)+1:tau1*d;
    idx2 = tau2*(d-1)+1:tau2*d;
    F2d = F2(idx2,idx2);
    F2d_kron = kron(eye(tau1),F2d);
    F2_kron = blkdiag(F2_kron,F2d_kron);
    L = blkdiag(L, kron(L1(idx1),L2(idx2,idx2)));
    Qc = blkdiag(Qc, kron(Qc1(d),L2(idx2,idx2)));
  end
  F = kron(F1,eye(tau2)) + F2_kron;
  H = kron(H1,[1 0]);
  Pinf = kron(Pinf1,eye(tau2));
  K = D*tau1; % model order
  % Solve the discrete-time state-space model (the function is from the EKF/UKF toolbox)
  [A,Q] = lti_disc(F,L,Qc,dt);
  
end