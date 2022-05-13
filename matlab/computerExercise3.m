%% Recursive Least Squares estimation

load 'tar2.dat';
load 'thx.dat'

na = 2;

subplot(2,1,1)
plot(tar2);
subplot(2,1,2)
plot(thx);

%% Recursive AR

lambda = 0.88; % Lower lambda yields noiser estimate as we are using fewer samples
X = recursiveAR(2);
X.ForgettingFactor = lambda;
X.InitialA = [1 0 0];

for kk = 1: length(tar2)
    [Aest(kk,:), yhat(kk)] = step(X, tar2(kk));
end

subplot(3,1,1)
plot(tar2);
subplot(3,1,2)
plot(thx);
subplot(3,1,3)
plot(Aest);

%% Choose the correct lambda with LS-estimate

n = 100;
lambda_line = linspace(0.85, 1, n);
ls2 = zeros(n,1);
yhat = zeros(n,1);
for i=1:length(lambda_line)
    reset(X);
    X.ForgettingFactor = lambda_line(i);
    X.InitialA = [1 0 0];
    for kk=1:length(tar2)
        [~, yhat(kk)] = step(X, tar2(kk));
    end
    ls2(i) = sum((tar2-yhat).^2);
end
plot(lambda_line, ls2);

lambdaOpt = lambda_line(ls2 == min(ls2)); %% Optimal lambda is = 0.94

%% Kalman filter

y = tar2;
N = length(y);

A = eye(2);
Re = [0.004 0; 0 0]; %Högt ger större varians
Rw = 1.25;

Rxx_1 = 10 * eye(2);
xtt_1 = [0 0]';

Xsave = zeros(2,N);
ehat = zeros(1,N);
yt1 = zeros(1,N);
yt2 = zeros(1,N);

for t=3:N    
    Ct = [-y(t-1) -y(t-2)];
    yhat(t) = Ct*xtt_1;
    ehat(t) = y(t) - yhat(t);
    
    % Update
    Ryy = Ct*Rxx_1*Ct'+ Rw;
    Kt = Rxx_1 * Ct'/Ryy;
    xt_t = xtt_1 + Kt*(ehat(t));
    Rxx = (eye(length(Rxx_1))-Kt*Ct)*Rxx_1;
    
    % Predict the next state
    xtt_1 = A*xt_t;
    Rxx_1 = A*Rxx*A'+Re;
    
    % Form 2-step prediction, ignore this at first. 
%     Ct1 = [];
%     yt1(t+1) = []; 
%     
%     Ct2 = [];
%     yt2(t+2) = []; 
%     
     Xsave(:,t) = xt_t;
end

figure
subplot(2, 1, 1)
plot(y)
subplot(2, 1,2)
plot(Xsave(1,:),'--r')
hold on
plot(Xsave(2,:),'--b')
plot(thx(:,1),'r')
plot(thx(:,2),'b')

% Increasing Re makes a much mor noisy random walk of the estimate, 
% (but also responds quite quickly??) 
 
% Increasing Rw makes the random walk smooth but slow. Good to combine?

% RLS can only handle AR, and in Kalman we have more liberty of setting
% some parameters as time-variant and time-invariant.

%% Sum of residual

sumSqError = sum(ehat.^2); %% 605 men det blir fel :/

%% 2.3 Using Kalman for prediction

rng(0);
N = 10000;
ee = 0.1*randn(N,1);
A0 = [1 -0.8 0.2];
y = filter(1, A0, ee);
Re = [1e-6 0; 0 1e-6];
Rw = 0.1; 

Rxx_1 = 10 * eye(2);
xtt_1 = [0 0]';

Xsave = zeros(2,N);
ehat = zeros(1,N);
yt1 = zeros(1,N);
yt2 = zeros(1,N);

for t=3:N-2    
    Ct = [-y(t-1) -y(t-2)];
    yhat(t) = Ct*xtt_1;
    ehat(t) = y(t) - yhat(t);
    
    % Update
    Ryy = Ct*Rxx_1*Ct'+ Rw;
    Kt = Rxx_1 * Ct'/Ryy;
    xt_t = xtt_1 + Kt*(ehat(t));
    Rxx = (eye(length(Rxx_1))-Kt*Ct)*Rxx_1;
    
    % Predict the next state
    xtt_1 = A*xt_t;
    Rxx_1 = A*Rxx*A'+Re;
    
    % Form 2-step prediction, ignore this at first. 
    Ct1 = [-y(t) -y(t-1)];
    yt1(t+1) = Ct1*xt_t; 
    
    Ct2 = [[-y(t+1) -y(t)]];
    yt2(t+2) = Ct2*xt_t; 
    
    Xsave(:,t) = xt_t;
end

figure
plot(y(end-100-2:end-2))
hold on
plot(yt1(end-100-1:end-1),'g')
plot(yt2(end-100:end),'r')
hold off
legend('y', 'k=1', 'k=2')

a2 = Xsave(2,10000-2) % a2 converges to 0.2092
sumSqError = sum(ehat(9800:end).^2); 

predError = yt2(980:end)-y(980-2:end-2)';
sumPredError = sum(predError.^2); 

% Increaing Rw --> higher sumSqError
% Increasing Re --> higher sumSqError but also when lowering

%% 2.4 Quality control of a process

N = 1000;
b = 20;
varE = 1;
varV = 4;

et = sqrt(varE).*randn(N+1,1);
vt = sqrt(varV).*randn(N,1);
ut = generateMarkov(N);

x = filter(1,[1 -1],et);
x = x(2:end);

y = x + b.*ut + vt;

%% Kalman filtering of the process

A = eye(2);
Re = [1 0; 0 1]; % How come the prediction changes?
Rw = 10; % Sätt som varians av data initialt

Rxx_1 = 10 * eye(2);
xtt_1 = [-5 20]';

Xsave = zeros(2,N);

for t=2:N   
    Ct = [1 ut(t)];
    yhat(t) = Ct*xtt_1;
    ehat(t) = y(t) - yhat(t);
    
    % Update
    Ryy = Ct*Rxx_1*Ct'+ Rw;
    Kt = Rxx_1 * Ct'/Ryy;
    xt_t = xtt_1 + Kt*(ehat(t));
    Rxx = (eye(length(Rxx_1))-Kt*Ct)*Rxx_1;
    
    % Predict the next state
    xtt_1 = A*xt_t;
    Rxx_1 = A*Rxx*A'+Re;
    
    % Save
    Xsave(:,t) = xt_t;
end

figure
plot(x, '-b');
hold on
yline(b,'-b');
plot(Xsave(1,:),'--r') % Our estimate of the state x
plot(Xsave(2,:),'--g') % Our estimate of b

sumSqError = sum(ehat.^2); 

%% Recursive temperature modeling

load 'svedala94.mat'
y = svedala94;
A6 = [1 0 0 0 0 0 -1];
ydiff = myFilter(A6,1,y);

T = linspace(datenum(1994,1,1),datenum(1994,12,31),length(ydiff));
plot(T, ydiff);
datetick('x');

%% Armax

th = armax(ydiff, [2 2]);
%   A(z) = 1 - 1.624 z^-1 + 0.6553 z^-2        
%                                              
%   C(z) = 1 - 0.8318 z^-1 - 0.1356 z^-2  
  
th_winter = armax(ydiff(1:540),[2 2]);
%   A(z) = 1 - 1.662 z^-1 + 0.7012 z^-2        
%                                              
%   C(z) = 1 - 0.8374 z^-1 - 0.1075 z^-2  
%   
th_summer = armax(ydiff(907:1458),[2 2]);
%   A(z) = 1 + 0.2509 z^-1 - 0.456 z^-2        
%                                              
%   C(z) = 1 + 1.019 z^-1 + 0.03068 z^-2  

%% Usage of recursiveARMA

X = recursiveARMA([2 2]);
X.InitialA = [1 th_winter.A(2:end)];
X.InitialC = [1 th_winter.C(2:end)];
X.ForgettingFactor = 0.99;
for k=1:length(ydiff)
    [Aest(k,:), Cest(k,:), yhat(k)] = step(X, ydiff(k));
end

subplot 311
sv = svedala94(7:end);
plot(T, sv);
datetick('x');
subplot 312
plot(Aest(:,2:3));
hold on
plot(repmat(th_winter.A(2:end), [length(ydiff) 1]), 'g:');
plot(repmat(th_summer.A(2:end), [length(ydiff) 1]), 'r:');
axis tight 
hold off
subplot 313
plot(Cest(:,2:3))
hold on
plot(repmat(th_winter.C(2:end), [length(ydiff) 1]), 'g:');
plot(repmat(th_summer.C(2:end), [length(ydiff) 1]), 'r:');
axis tight 
hold off

%% 2.6 Recursive temperature modeling, again

load 'svedala94.mat'
y = svedala94(850:1100);
y = y - mean(y);
plot(y);

%% Data for entire year

load 'svedala94.mat'
y = svedala94;
y = y-y(1);

%% Sinusoidal input 

S = 48;

t = (1:length(y))';
U = [sin(2*pi*t/S) cos(2*pi*t/S)];
Z = iddata(y, U);
model = [ 3 [ 1 1 ] 4 [ 0 0 ] ] ;

%[ na [ nb_1 nb_2 ] nc [ nk_1 nk_2 ] ] ;
thx = armax(Z,model);

plot(y);
hold on 
plot(U*cell2mat(thx.b)','-r');

%% 

U = [sin(2*pi*t/S) cos(2*pi*t/S) ones(size(t))];
Z = iddata(y,U);
m0 = [thx.A(2:end) cell2mat(thx.B) 0 thx.C(2:end)];
Re = diag ( [ 0 0 0 0 0 1 0 0 0 0 ] ) ; % How does this impact the results?
model = [3 [ 1 1 1 ] 4 0 [ 0 0 0 ] [ 1 1 1 ] ] ;
% [na [nb1 · · · ] nc nd [nf1 · · · ] [nk1 · · · ]]

[thr, yhat] = rpem (Z , model , 'kf', Re, m0);

%% Plot it

figure
plot(y)
hold on
m = thr(:,S);
a = thr(end,4);
b = thr(end,5);
y_mean = m + a*U(:,1) + b*U(:,2);
y_mean = [0;y_mean(1:end-1)];
plot(y_mean,'-r')

%% Yhat vs y

figure
plot(yhat);
hold on
plot(y)











