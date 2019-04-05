clear;
% sampling from a Gaussian process posterior

  var = 1.; % variance (sigma^2)
  len = 50; % lengthscale (smoothness)
  kernel = 'se'; % 'exp', 'matern32' or 'se' 
  T = 200; % number of test locations
  maxT = 1000;
  t_star = linspace(1,maxT,T)';
  dt = maxT/T; % time-step size (if this is too small it can lead to numerical problems)
  % add new data points and locations here:
  train_loc =   [11   20    35   42 111  120 129 135 142  150 170]; % training locations
  y         =   [1  -1.4 -0.25 0.75 0.8  0.2  -1  -1 0.2 -1.3 0.9]'; % generate some toy data
  t = t_star(train_loc);
  var_y = .05; % observation noise
  blue = [0.4 0.3 0.7];
  
  
  
  
%% Kernel-based posterior %%
tic; disp('kernel-based sampling');
  K = cov(t, t, kernel, len, var) + var_y * eye(length(y)); % prior covariance of the data
  K_star = cov(t, t_star, kernel, len, var); % prior joint covariance of data and test locations
  K_star_star = cov(t_star, t_star, kernel, len, var); % prior covariance of test locations

  post_mean = K_star' * inv(K) * y; % posterior mean of data and test locations
  post_cov = K_star_star - K_star' * inv(K) * K_star; % posterior covariance
  log_lik_kern = (-1/2)*y'*inv(K+var_y*eye(size(K)))*y - (1/2)*log(det(K+var_y*eye(size(K)))) - (length(y)/2)*log(2*pi); % log marginal likelihood
  % generate 3 samples from the posterior distribution
  x_k = mvnrnd(post_mean,post_cov,3);
toc  
  figure(1);
  subplot(2,1,1); cla();
  plot(t,y,'k.','MarkerSize',10)
  hold on
  plot(t_star,post_mean,'k--')
  p1=patch([t_star',fliplr(t_star')],[(post_mean-1.96.*sqrt(diag(post_cov)))',fliplr((post_mean+1.96.*sqrt(diag(post_cov)))')],blue);
  p1.EdgeAlpha=0; alpha(p1,0.3);
  plot(t_star,x_k)
  ylim([-2.5, 2.5])
  legend('Observed data','posterior mean','95% confidence','posterior samples')
  title('Kernel-based posterior')


  
%% State space posterior %%
tic; disp('state-space sampling');
  cf_to_ss = str2func(strcat('cf_',kernel,'_to_ss'));
  [F,L,Qc,H,Pinf] = cf_to_ss(var, len); % calculate state-space model
  
  [A,Q] = lti_disc(F,L,Qc,dt); % discretise the model
  
  y_kf = NaN*t_star; y_kf(train_loc) = y;
  [log_lik_ss,post_mean_ss,post_cov_ss] = kalmanSmoother(A,Q,H,Pinf,var_y,y_kf);
  % generate 3 samples from the posterior distribution
  x_s = ss_prior_samp(A,Q,H,Pinf,T,3); % sample from the prior
  x_s_post = post_mean_ss + sqrt(post_cov_ss) .* x_s / sqrt(var); % inspired by the reparametrisation trick
toc  
  figure(1);
  subplot(2,1,2); cla();
  plot(t,y,'k.','MarkerSize',10)
  hold on
  plot(t_star,post_mean_ss,'k--')
  p1=patch([t_star',fliplr(t_star')],[(post_mean_ss-1.96.*sqrt(post_cov_ss))',fliplr((post_mean_ss+1.96.*sqrt(post_cov_ss))')],blue);
  p1.EdgeAlpha=0; alpha(p1,0.3);
  plot(t_star,x_s_post)
  ylim([-2.5, 2.5])
  title('State space posterior')
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  
%% functions
  
  % kernel based covariance calculation
  function K = cov(t1, t2, kern, l, v)
    K = zeros(length(t1), length(t2));
    for i=1:length(t1)
      for j=1:length(t2)
          r = abs(t1(i) - t2(j));
          if strcmp(kern,'exp')
            K(i, j) = v * exp(-r / l);
          elseif strcmp(kern,'matern32')
            K(i, j) = v * (1 + sqrt(3)*r/l) * exp((-sqrt(3)) * r / l);
          elseif strcmp(kern,'se')
            K(i, j) = v * exp((-1/2) * r^2 / l^2);
          end
      end
    end
  end
  
  
  
  % state space sampling
  function samp = ss_prior_samp(A,Q,H,Pinf,T,num_samps)
    samp = zeros(T,num_samps);
    for s=1:num_samps
      z = zeros(length(H),T);
      z(:,1) = chol(Pinf)' * randn(length(H),1);
      samp(1,s) = H * z(:,1);
      for k=2:T
        z(:,k) = A * z(:,k-1) + chol(Q)' * randn(length(H),1);
        samp(k,s) = H * z(:,k);
      end
    end
  end
  
  
  
  % Kalman filter and smoother for state space posterior calc
  function  [lik,Xfin,Pfin] = kalmanSmoother(A,Q,H,P,vary,y)
    T = length(y);
    Y = reshape(y,[1,1,T]);
    lik=0;
    if length(vary) == 1
      vary = vary * ones(T,1);
    end
    m = zeros(size(A,1),1);
    MS = zeros(size(m,1),size(Y,3));
    PS = zeros(size(m,1),size(m,1),size(Y,3));
    Pfin = zeros(size(Y,3),sum(H));
    % ### Forward filter
    for k=1:T
        R = vary(k);
        % Prediction
        if (k>1)
            m = A*m;
            P = A*P*A' + Q;
        end
        % Kalman update
        if ~isnan(Y(:,:,k))
          S = H*P*H' + R;
          K = P*H'/S;
          v = Y(:,:,k)-H*m;
          m = m + K*v;
          P = P - K*H*P;
          % Evaluate the energy (neg. log lik): Check this
          lik = lik + .5*size(S,1)*log(2*pi) + .5*log(S) + .5*v'/S*v;
        end
        PS(:,:,k) = P;
        MS(:,k)   = m;
    end
    % ### Backward smoother
    for k=size(MS,2)-1:-1:1
        % Smoothing step (using Cholesky for stability)
        PSk = PS(:,:,k);
        % Pseudo-prediction
        PSkp = A*PSk*A'+Q;
        [L,~] = chol(PSkp,'lower'); % Solve the Cholesky factorization
        % Continue smoothing step
        G = PSk*A'/L'/L;
        % Do update
        m = MS(:,k) + G*(m-A*MS(:,k));
        P = PSk + G*(P-PSkp)*G';
        MS(:,k)   = m;
        PS(:,:,k) = P;
        Pfin(k,:) = diag(P(find(H),find(H)));
    end
    lik = -lik;
    Xfin = reshape(MS,[1 size(MS)]);
    Xfin = squeeze(Xfin(1,find(H),:));
  end
  
  
  
  
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