clear;
% sampling from the time-frequency model posterior
  
  % parameters of the 3 periodic components / filters
  var =   [0.06  0.12  0.25]; % variance (sigma^2)
  len =   [60     30    100]; % lengthscale (smoothness)
  omega = [pi/4 pi/8  pi/12]; % centre frequency
  
  dt = 1; % time-step size
  kernel = 'matern32'; % 'exp', 'matern32' or 'se' 
  T = 200;
  t_star = linspace(1,T,T)';
  train_loc =   [1:75, 126:200]; % training locations
  test_loc  =   [76:125];
  var_y = 0.02; % observation noise
    K_test = cov(t_star,t_star,kernel,len,var,omega);
    y_test = mvnrnd(zeros(1,size(K_test,1)),K_test)' + sqrt(var_y)*randn(T,1);
  y = y_test(train_loc); % generate some toy data
  t = t_star(train_loc);
  red = [0.7 0.3 0.4];blue = [0.4 0.3 0.7];green = [0.4 0.7 0.3];
  
  
  
  
%% Kernel-based posterior %%
tic; disp('kernel-based sampling');
  K = cov(t, t, kernel, len, var, omega) + var_y * eye(length(y)); % prior covariance of the data
  K_star = cov(t, t_star, kernel, len, var, omega); % prior joint covariance of data and test locations
  K_star_star = cov(t_star, t_star, kernel, len, var, omega); % prior covariance of test locations

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
  plot(test_loc,x_k(:,test_loc))
  ylim([-2.5, 2.5])
  legend('Observed data','posterior mean','95% confidence','posterior samples')
  title('Kernel-based posterior')  
  


  
  
%% State space posterior %%
tic; disp('state-space sampling');
  D = length(omega);
  [A,Q,H,Pinf,K,tau1] = get_disc_model(len,var,omega,D,kernel);
  
  y_kf = NaN*t_star; y_kf(train_loc) = y;
  [log_lik_ss,post_mean_ss_,post_cov_ss_] = kalmanSmoother(A,Q,H,Pinf,var_y,y_kf);
  post_mean_ss = sum(post_mean_ss_,2);
  post_cov_ss = sum(post_cov_ss_,2); post_cov_ss(t) = post_cov_ss(t)/length(var);
  % generate 3 samples from the posterior distribution
  x_s = ss_prior_samp(A,Q,H,Pinf,T,3); % sample from the prior
  x_s_post = post_mean_ss + sqrt(post_cov_ss) .* x_s / sqrt(sum(var)); % inspired by the reparametrisation trick
toc
  figure(1);
  subplot(2,1,2); cla();
  plot(t,y,'k.','MarkerSize',10)
  hold on
  plot(t_star,post_mean_ss,'k--')
  p1=patch([t_star',fliplr(t_star')],[(post_mean_ss-1.96.*sqrt(post_cov_ss))',fliplr((post_mean_ss+1.96.*sqrt(post_cov_ss))')],blue);
  p1.EdgeAlpha=0; alpha(p1,0.3);
  plot(test_loc,x_s_post(test_loc,:))
  ylim([-2.5, 2.5])
  title('State space posterior')

  
  
  
  
%% Extract frequency components
  
  figure(2);clf;
  subplot(5,1,1)
  plot(y_kf,'k','LineWidth',1.8)
  title('Observed signal')
  subplot(5,1,2)
  plot(post_mean_ss_(:,1),'Color',red,'LineWidth',1.8)
  p1=patch([t_star',fliplr(t_star')],[(post_mean_ss_(:,1)-1.96.*sqrt(post_cov_ss_(:,1)))',fliplr((post_mean_ss_(:,1)+1.96.*sqrt(post_cov_ss_(:,1)))')],red);
  p1.EdgeAlpha=0; alpha(p1,0.3);
  ylim([-1.2,1.2])
  title('First periodic component')
  subplot(5,1,3)
  plot(post_mean_ss_(:,2),'Color',green,'LineWidth',1.8)
  p2=patch([t_star',fliplr(t_star')],[(post_mean_ss_(:,2)-1.96.*sqrt(post_cov_ss_(:,2)))',fliplr((post_mean_ss_(:,2)+1.96.*sqrt(post_cov_ss_(:,2)))')],green);
  p2.EdgeAlpha=0; alpha(p2,0.3);
  ylim([-1.2,1.2])
  title('Second periodic component')
  subplot(5,1,4)
  plot(post_mean_ss_(:,3),'Color',blue,'LineWidth',1.8)
  p3=patch([t_star',fliplr(t_star')],[(post_mean_ss_(:,3)-1.96.*sqrt(post_cov_ss_(:,3)))',fliplr((post_mean_ss_(:,3)+1.96.*sqrt(post_cov_ss_(:,3)))')],blue);
  p3.EdgeAlpha=0; alpha(p3,0.3);
  ylim([-1.2,1.2])
  title('Third periodic component')
  subplot(5,1,5)
  imagesc(log(abs(hilbert(post_mean_ss_))'.^2))
  title('Spectrogram')
  
  
  
  
  
  
  
  
  
  
  
  
  
  

%% functions

  % kernel based covariance calculation
  function K = cov(t1, t2, kern, l, v, om)
    K = zeros(length(t1), length(t2));
    for i=1:length(t1)
      for j=1:length(t2)
          r = abs(t1(i) - t2(j));
          if strcmp(kern,'exp')
            K(i, j) = sum(v .* cos(om*r) .* exp(-r ./ l));
          elseif strcmp(kern,'matern32')
            K(i, j) = sum(v .* cos(om*r) .* (1 + sqrt(3)*r./l) .* exp((-sqrt(3)) * r ./ l));
          elseif strcmp(kern,'se')
            K(i, j) = sum(v .* cos(om*r) .* exp((-1/2) * r^2 ./ l.^2));
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
    Xfin = squeeze(Xfin(1,find(H),:))';
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
