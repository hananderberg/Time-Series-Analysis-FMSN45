%% Kalman filter

close all;
clc;

load('/Users/Hanna/Documents/MATLAB/Matlab_tsa_21/project/proj21data.mat')

B1 = MboxJ.B(1); 
B2 = MboxJ.B(2); 
A21 = MboxJ.F(1);
A22 = MboxJ.F(2);
C1 = MboxJ.C;
A1 = MboxJ.D;

y = waterflow(120:1096);
x1 = nbd(120:1096);
x1(x1 == -1) = NaN;
x1 = fillmissing(x1,'linear');
x2 = temp(120:1096);
N = length(y);

k = 7;
y = y_test;

%% Predict rain ARMA(3,1)

%Estimate the unknown parameters using a Kalman filter and form the k-step prediction.

%x1 = x1_test;
x1(x1 == -1) = NaN;
x1 = fillmissing(x1,'linear');
N = length(x1);
p0 = 3;                                         % Number of unknowns in the A polynomial.
q0 = 1;                                         % Number of unknowns in the C polynomial.

A     = eye(p0+q0);
Rw    = 1.25;                                      % Measurement noise covariance matrix, R_w
Re    = 1e-4*eye(p0+q0);                                   % System noise covariance matrix, R_e
Rx_t1 = eye(p0+q0);                             % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(p0+q0,N);                         % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N,1);                             % Estimated output.
yhatk_rain = zeros(N,1);                             % Estimated k-step prediction.

xt(:,2) = [foundModelNbd.A(foundModelNbd.A~=1) foundModelNbd.C(foundModelNbd.C~=1)]';       %Set initial states
yk_ = zeros(N,1);

for t=4:N-k                                     % We use t-3, so start at t=4. As we form a k-step prediction, end the loop at N-k.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ -x1(t-1) -x1(t-2) -x1(t-3) h_et(t-1) ];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = x1(t)-yhat(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    
    %We have an AR 3
    yk_(1) = x1(t-1);
    yk_(2) = x1(t);
    
    Ck = [ -x1(t) -x1(t-1) -x1(t-2) h_et(t) ];           % C_{t+1|t}
    yk_(3) = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}

    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    
    for k0=2:k
        Ck = [ -yk_(k0+1) -yk_(k0) -yk_(k0-1) h_et(t+k0-1) ]; % C_{t+k|t}
        yk_(k0+2) = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
    end
    yhatk_rain(t+k) = yk_(k+2);
end

figure
plot(x1);
hold on
plot(yhatk_rain);

figure
Q0 = [foundModelNbd.A(foundModelNbd.A~=1) foundModelNbd.C(foundModelNbd.C~=1)];                       % These are the true parameters we seek.
plot( xt' )
legend
for k0=1:length(Q0)
    line([1 N], [Q0(k0) Q0(k0)], 'Color','red','LineStyle',':')
end
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re, Rw))
xlabel('Time')
ylim([-1.5 1.5])
xlim([1 N-k])

%% Predict temperature ARMA(2,1)

%Estimate the unknown parameters using a Kalman filter and form the k-step prediction.

x2 = temp(120:1096);
%x2 = x2_test;

N = length(x2);
p0 = 2;                                         % Number of unknowns in the A polynomial.
q0 = 1;                                         % Number of unknowns in the C polynomial.

A     = eye(p0+q0);
Rw    = 1.25;                                      % Measurement noise covariance matrix, R_w
Re    = 1e-4*eye(p0+q0);                                   % System noise covariance matrix, R_e
Rx_t1 = eye(p0+q0);                             % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(p0+q0,N);                         % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N,1);                             % Estimated output.
yhatk_temp = zeros(N,1);                             % Estimated k-step prediction.

xt(:,2) = [foundModelTemp.A(foundModelTemp.A~=1) foundModelTemp.C(foundModelTemp.C~=1)]';       %Set initial states
yk_ = zeros(N,1);

for t=4:N-k                                     % We use t-3, so start at t=4. As we form a k-step prediction, end the loop at N-k.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ -x2(t-1) -x2(t-2) h_et(t-1) ];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = x2(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    yk_(1) = x2(t);
    
    Ck = [ -x2(t) -x2(t-1) h_et(t) ];           % C_{t+1|t}
    yk_(2) = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
    
    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    for k0=2:k
        Ck = [ -yk_(k0) -yk_(k0-1) h_et(t+k0-1) ]; % C_{t+k|t}
        yk_(k0+1) = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
    end
    yhatk_temp(t+k) = yk_(k+1);                            % Note that this should be stored at t+k.
end

figure
plot(x2);
hold on
plot(yhatk_temp);

figure
Q0 = [foundModelTemp.A(foundModelTemp.A~=1) foundModelTemp.C(foundModelTemp.C~=1)];                       % These are the true parameters we seek.
plot( xt' )
legend
for k0=1:length(Q0)
    line([1 N], [Q0(k0) Q0(k0)], 'Color','red','LineStyle',':')
end
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re, Rw))
xlabel('Time')
ylim([-1.5 1.5])
xlim([1 N-k])


%% Estimate the unknown parameters using a Kalman filter and form the k-step prediction.

N = length(y);

p0 = length(KA)-1; %number of unknowns in KA
q0 = length(KC)-1; %number of unknowns in KC
r0 = length(KB1); %number of unknowns in KB1
s0 = length(KB2); %number of unknowns in KB2

sumOfUnknowns = p0 + r0 + q0 + s0;

A     = eye(sumOfUnknowns);
Rw    = 10;                                      % Measurement noise covariance matrix, R_w
Re    = 1e-6;                                   % System noise covariance matrix, R_e
Rx_t1 = eye(sumOfUnknowns);                     % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(sumOfUnknowns,N);                 % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N,1);                             % Estimated output.
yhatk = zeros(N,1);                             % Estimated k-step prediction.

xt(:,2) = [KA(KA~=1) KC(KC~=1) KB1 KB2]';       %Set initial states
%xt(:,2) = [-0.8752 -0.0852  -0.9525 0.8337 0.0812 -0.3518 0.3079 0.0300]';       %Set initial states

KA_shifts = find(KA~=0)-1;
KA_shifts = KA_shifts(2:end);
KC_shifts = find(KC~=0)-1;
KB1_shifts = find(KB1~=0)-1;
KB2_shifts = find(KB2~=0)-1;

yk_ = zeros(N,1);

for t=3:N-k                                     % We use t-2, so start at t=3. As we form a k-step prediction, end the loop at N-k.
    
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                        % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ -y(t-KA_shifts)' -yhatk_rain(t-KB1_shifts)' -yhatk_temp(t-KB2_shifts)'];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = y(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 
    
    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re
   
    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    yk_(1) = y(t);
    
    Ck = [ -y(t+1-KA_shifts)' -yhatk_rain(t+1-KB1_shifts)' -yhatk_temp(t+1-KB2_shifts)' ];           % C_{t+1|t}
    yk_(2) = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
    
    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    for k0=2:k
        Ck    = [ -yk_(k0) -yk_(k0-1) -yhatk_rain(t+k0-KB1_shifts)' -yhatk_temp(t+k0-KB2_shifts)'];     % C_{t|t-1}
        yk_(k0+1) = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
    end
    yhatk(t+k) = yk_(k+1);                            % Note that this should be stored at t+k.
end


%% Examine the estimated parameters.

figure
Q0 = [KA(2:end) KB1 KB2];                       % These are the true parameters we seek.
plot( xt' )
legend
for k0=1:length(Q0)
    line([1 N], [Q0(k0) Q0(k0)], 'Color','red','LineStyle',':')
end
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re, Rw))
xlabel('Time')
ylim([-1.5 1.5])
xlim([1 N-k])


%% Show the one-step prediction. 
figure
plot( [y yhat] )
title('One-step prediction using the Kalman filter')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 N-k])


%% Show the k-step prediction. 
figure
plot( [y(1:N-k) yhatk(k+1:N)] )
title( sprintf('%i-step prediction using the Kalman filter (shifted %i steps)', k, k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 N-k])


%% Examine k-step prediction residual.
ey = y-yhatk;
ey = ey(N-200:N-k);                             % Ignore the initial values to let the filter converge first.
plotACFnPACF( ey, 40, sprintf('%i-step prediction using the Kalman filter', k)  );

variance = var(ey)

%%%%%%%%%%%%%%%%%%%%
%% Eliminate parameters one by one with model data

y = waterflow(120:1096);
N = length(y);

% Start with A1
p0 = length(KA)-1; %number of unknowns in KA
q0 = length(KC)-1; %number of unknowns in KC
r0 = length(KB1); %number of unknowns in KB1
s0 = length(KB2); %number of unknowns in KB2

sumOfUnknowns = p0 + r0 + q0 + s0;

A     = eye(sumOfUnknowns-1);
Rw    = 4;                                      % Measurement noise covariance matrix, R_w
Re    = 1e-6;                                   % System noise covariance matrix, R_e
Rx_t1 = eye(sumOfUnknowns-1);                     % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(sumOfUnknowns,N);                 % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N,1);                             % Estimated output.
yhatk = zeros(N,1);                             % Estimated k-step prediction.

%xt(:,2) = [-0.8752 -0.0852 -0.9525 0.8337 0.0812 -0.3518 0.3079 0.0300]';       %Set initial states
xt(:,2) = [-0.8752 -0.0852 -0.9525 0.8337 0.0812 -0.3518 0.3079 0.0300]';       %Set initial states

% KA, KB1, KB2

KA_shifts = find(KA~=0)-1;
KA_shifts = KA_shifts(2:end);
KC_shifts = find(KC~=0)-1;
KB1_shifts = find(KB1~=0)-1;
KB2_shifts = find(KB2~=0)-1;

variances = zeros(sumOfUnknowns, 1);

for s=1:sumOfUnknowns
    xt    = zeros(sumOfUnknowns,N);
    xt(:,2) = [-0.8752 -0.0852 -0.9525 0.8337 0.0812 -0.3518 0.3079 0.0300]';       %Set initial states
    xt(s,:) = [];
    
    yk_ = zeros(N,1);
    
    for t=3:N-k                                     % We use t-2, so start at t=3. As we form a k-step prediction, end the loop at N-k.
        
        % Update the predicted state and the time-varying state vector.
        x_t1 = A*xt(:,t-1);                        % x_{t|t-1} = A x_{t-1|t-1}
        C    = [ -y(t-KA_shifts)' -yhatk_rain(t-KB1_shifts)' -yhatk_temp(t-KB2_shifts)'];     % C_{t|t-1}
        C(s) = [];
        
        % Update the parameter estimates.
        Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
        Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
        yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
        h_et(t) = y(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
        xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} )
        
        % Update the covariance matrix estimates.
        Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
        Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re
        
        % Form the k-step prediction by first constructing the future C vector
        % and the one-step prediction. Note that this is not yhat(t) above, as
        % this is \hat{y}_{t|t-1}.
        yk_(1) = y(t);
        
        Ck = [ -y(t+1-KA_shifts)' -yhatk_rain(t+1-KB1_shifts)' -yhatk_temp(t+1-KB2_shifts)' ];  
        Ck(s) = [];
        yk_(2) = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
        
        % Note that the k-step predictions is formed using the k-1, k-2, ...
        % predictions, with the predicted future noises being set to zero. If
        % the ARMA has a higher order AR part, one needs to keep track of each
        % of the earlier predicted values.
        for k0=2:k
            Ck    = [ -yk_(k0) -yk_(k0-1) -yhatk_rain(t+k0-KB1_shifts)' -yhatk_temp(t+k0-KB2_shifts)'];
            Ck(s) = [];
            yk_(k0+1) = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        end
        yhatk(t+k) = yk_(k+1);                            % Note that this should be stored at t+k.
    end

     ey = y-yhatk;
     ey = ey(N-200:N-k);                             % Ignore the initial values to let the filter converge first.
     
     variances(s) = var(ey);
    
end

