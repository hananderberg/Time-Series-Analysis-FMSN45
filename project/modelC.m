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


%% Estimate the unknown parameters using a Kalman filter and form the k-step prediction.

k  = 4;                                         % k-step prediction.
% y = y_validation;
% x1 = x1_validation;
% x2 = x2_validation;

p0 = length(KA)-1; %number of unknowns in KA
q0 = length(KC)-1; %number of unknowns in KC
r0 = length(KB1); %number of unknowns in KB1
s0 = length(KB2); %number of unknowns in KB2

sumOfUnknowns = p0 + r0 + q0 + s0;

A     = eye(sumOfUnknowns);
Rw    = 4;                                      % Measurement noise covariance matrix, R_w
Re    = 1e-6;                                   % System noise covariance matrix, R_e
Rx_t1 = eye(sumOfUnknowns);                     % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(sumOfUnknowns,N);                 % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N,1);                             % Estimated output.
yhatk = zeros(N,1);                             % Estimated k-step prediction.

xt(:,2) = [KA(KA~=1) KC(KC~=1) KB1 KB2]';       %Set initial states

KA_shifts = find(KA~=0)-1;
KA_shifts = KA_shifts(2:end);
KC_shifts = find(KC~=0)-1;
KB1_shifts = find(KB1~=0)-1;
KB2_shifts = find(KB2~=0)-1;

for t=3:N-k                                     % We use t-2, so start at t=3. As we form a k-step prediction, end the loop at N-k.
        
        %%%%% Predict rain input %%%%%%
        N_rain = length(x1);
        p0_rain = 3;                                         % Number of unknowns in the A polynomial.
        q0_rain = 1;                                         % Number of unknowns in the C polynomial.

        A_rain     = eye(p0_rain+q0_rain);
        Rw_rain    = 1.25;                                      % Measurement noise covariance matrix, R_w
        Re_rain    = 1e-4*eye(p0_rain+q0_rain);                                   % System noise covariance matrix, R_e
        Rx_t1_rain = eye(p0_rain+q0_rain);                             % Initial covariance matrix, R_{1|0}^{x,x}
        h_et_rain  = zeros(N_rain,1);                             % Estimated one-step prediction error.
        xt_rain    = zeros(p0_rain+q0_rain, N_rain);                         % Estimated states. Intial state, x_{1|0} = 0.
        yhat_rain  = zeros(N_rain,1);                             % Estimated output.
        yhatk_rain = zeros(N_rain,1);                             % Estimated k-step prediction.
        
        xt_rain(:,2) = [foundModelNbd.A(foundModelNbd.A~=1) foundModelNbd.C(foundModelNbd.C~=1)]';       %Set initial states

        for m=4:N_rain-k                                     % We use t-3, so start at t=4. As we form a k-step prediction, end the loop at N-k.
            % Update the predicted state and the time-varying state vector.
            x_t1 = A_rain*xt_rain(:,m-1);                         % x_{t|t-1} = A x_{t-1|t-1}
            C    = [ -x1(m-1) -x1(m-2) -x1(m-3) h_et_rain(m-1) ];     % C_{t|t-1}

            % Update the parameter estimates.
            Ry = C*Rx_t1_rain*C' + Rw_rain;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
            Kt = Rx_t1_rain*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
            yhat(m) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
            h_et(m) = x1(m)-yhat_rain(m);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
            xt_rain(:,m) = x_t1 + Kt*( h_et_rain(m) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

          % Update the covariance matrix estimates.
            Rx_t  = Rx_t1_rain - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
            Rx_t1_rain = A_rain*Rx_t*A_rain' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

            % Form the k-step prediction by first constructing the future C vector
            % and the one-step prediction. Note that this is not yhat(t) above, as
            % this is \hat{y}_{t|t-1}.
            Ck = [ -x1(m) -x1(m-1) -x1(m-2) h_et_rain(m) ];           % C_{t+1|t}
            yk = Ck*xt_rain(:,m);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}

            % Note that the k-step predictions is formed using the k-1, k-2, ...
            % predictions, with the predicted future noises being set to zero. If
            % the ARMA has a higher order AR part, one needs to keep track of each
            % of the earlier predicted values.
            for k0=2:k
                Ck = [ -yk -x1(m+k0-2) -x1(m+k0-3) h_et(m+k0-1) ]; % C_{t+k|t}
                yk = Ck*A_rain^k*xt_rain(:,m);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
            end
            yhatk_rain(m+k) = yk;                            % Note that this should be stored at t+k.
        end
        
        %%%%% Predict temperature %%%%%%%%%%
        N_temp = length(x2);                                         
        p0_temp = 2;                                         % Number of unknowns in the A polynomial.
        q0_temp = 1;                                         % Number of unknowns in the C polynomial.
        
        A_temp = eye(p0_temp+q0_temp);
        Rw_temp    = 1.25;                                      % Measurement noise covariance matrix, R_w
        Re_temp    = 1e-4*eye(p0_temp+q0_temp);                                   % System noise covariance matrix, R_e
        Rx_t1_temp = eye(p0_temp+q0_temp);                             % Initial covariance matrix, R_{1|0}^{x,x}
        h_et_temp  = zeros(N_temp,1);                             % Estimated one-step prediction error.
        xt_temp    = zeros(p0_temp+q0_temp,N_temp);                         % Estimated states. Intial state, x_{1|0} = 0.
        yhat_temp  = zeros(N_temp,1);                             % Estimated output.
        yhatk_temp = zeros(N_temp,1);                             % Estimated k-step prediction.
        
        xt_temp(:,2) = [foundModelTemp.A(foundModelTemp.A~=1) foundModelTemp.C(foundModelTemp.C~=1)]';       %Set initial states
        
        for l=3:N-k                                     % We use t-3, so start at t=4. As we form a k-step prediction, end the loop at N-k.
            % Update the predicted state and the time-varying state vector.
            x_t1 = A_temp*xt_temp(:,l-1);                         % x_{t|t-1} = A x_{t-1|t-1}
            C    = [ -x2(l-1) -x2(l-2) h_et(l-1) ];     % C_{t|t-1}
            
            % Update the parameter estimates.
            Ry = C*Rx_t1_temp*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
            Kt = Rx_t1_temp*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
            yhat_temp(l) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
            h_et(l) = x2(l)-yhat(l);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
            xt_temp(:,l) = x_t1 + Kt*( h_et(l) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} )
            
            % Update the covariance matrix estimates.
            Rx_t  = Rx_t1_temp - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
            Rx_t1_temp = A_temp*Rx_t*A_temp' + Re_temp;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re
            
            % Form the k-step prediction by first constructing the future C vector
            % and the one-step prediction. Note that this is not yhat(t) above, as
            % this is \hat{y}_{t|t-1}.
            Ck = [ -x2(l) -x2(l-1) h_et_temp(l) ];           % C_{t+1|t}
            yk = Ck*xt_temp(:,l);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
            
            % Note that the k-step predictions is formed using the k-1, k-2, ...
            % predictions, with the predicted future noises being set to zero. If
            % the ARMA has a higher order AR part, one needs to keep track of each
            % of the earlier predicted values.
            for k0=2:k
                Ck = [ -yk -x2(l+k0-2) h_et_temp(l+k0-1) ]; % C_{t+k|t}
                yk = Ck*A_temp^k*xt_temp(:,l);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
            end
            yhatk_temp(l+k) = yk;                            % Note that this should be stored at t+k.
        end
    
    %%%%%%%% ACTUAL KALMAN LOOP %%%%%%%%
    
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
    Ck = [ -y(t+1-KA_shifts)' -yhatk_rain(t+1-KB1_shifts)' -yhatk_temp(t+1-KB2_shifts)' ];           % C_{t+1|t}
    yk = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
    
    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    for k0=2:k
        Ck    = [ -y(t+k0-KA_shifts)' -yhatk_rain(t+k0-KB1_shifts)' -yhatk_temp(t+k0-KB2_shifts)'];     % C_{t|t-1}
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
    end
    yhatk(t+k) = yk;                            % Note that this should be stored at t+k.
end

%Show the k-step prediction for rain. 
figure
plot( [x1(1:N-k) yhatk_rain(k+1:N)] )
title( sprintf('%i-step prediction using the Kalman filter (shifted %i steps)', k, k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 N-k])

%Show the k-step prediction for temperature. 
figure
plot( [x2(1:N-k) yhatk_temp(k+1:N)] )
title( sprintf('%i-step prediction using the Kalman filter (shifted %i steps)', k, k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 N-k])

%Show the k-step prediction for waterflow. 
figure
plot( [y(1:N-k) yhatk(k+1:N)] )
title( sprintf('%i-step prediction using the Kalman filter (shifted %i steps)', k, k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 N-k])

