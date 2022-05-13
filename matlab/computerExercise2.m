%% Generating some data following the Box-Jenkins model

rng(0);
n = 500;
omitted = 100;

A3 = [1 0.5];
C3 = [1 -0.3 0.2];
w = sqrt(2)*randn(n+omitted,1);
x = filter(C3, A3, w);

A1 = [1 -0.65];
A2 = [1 0.9 0.78];
C = 1;
B = [0 0 0 0 0.4];
e = sqrt(1.5)*randn(n + omitted,1);

y = filter(C,A1,e) + filter(B,A2,x);
x = x(omitted+1:end); y = y(omitted+1:end);
clear A1 A2 C B e w A3 C3

%% Make xt white

plotACFnPACF(x, 20, 'Analysis of x');
foundModel = estimateARMA(x, [1 1], [1 1 1], ' ARMA(1,2) ', 20);

%% Find w_t and eps_t

w_t = filter( foundModel.A, foundModel.C, x );   
w_t = w_t(length(foundModel.A):end );

eps_t = filter( foundModel.A, foundModel.C, y );   
eps_t = eps_t(length(foundModel.A):end );

%% CCF from w_t to eps_t

M = 40;
stem(-M:M,crosscorr(w_t, eps_t,M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% Find A2 and B

A2 = [1 0 0]; %since r = 2
B = [0 0 0 0 1]; %since d = 4, s = 0
Mi = idpoly([1], [B], [], [], [A2]);
z = iddata(y,x);
Mba2 = pem(z, Mi); 
present(Mba2);
etilde = resid(Mba2, z).y;
plotBasics(etilde); %etilde is not white noise

%% Calculate the residual etilde and verify it is uncorrelated with xt

stem(-M:M,crosscorr(etilde, x ,M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% Find C1 and A1

% The function call is estimateBJ( y, x, C1, A1, B, A2, titleStr, noLags )
C1 = [1];
A1 = [1 1] 

[foundModel, ey] = estimateBJ(y, x, C1, A1, Mba2.B, Mba2.F, 'BJ model 1', 20);

%% Crosscorrelation between x and etilde

etilde = y - filter( foundModel.B, foundModel.F, x );

etilde  = etilde(length(foundModel.B):end );
filter_xt = x(length(foundModel.B):end );

stem(-M:M,crosscorr(etilde, filter_xt ,M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% Estimate all polynomials again using pem

A1 =[1 0];
A2 = [1 0 0];
B = [1 0 0 0 0];
C = 1;
Mi = idpoly(1,B,C,A1,A2);
z = iddata(y,x);
MboxJ = pem(z,Mi);
present(MboxJ);
ehat = resid(MboxJ,z).y;

M = 40;
stem(-M:M,crosscorr(x,ehat, M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% Hairdryer data

load ('tork.dat');
sampDist = 0.08;
n = 300;
tork = tork - repmat(mean(tork), length(tork), 1);
y = tork(:,1); x = tork(:,2);
z = iddata(y,x, sampDist);
plot(z(1:300));

%% Make input white for hairdryer data

%plotACFnPACF(x, 20, 'Analysis of input');
foundModel = estimateARMA(x, [1 1], [1], ' AR(1) ', 20);

%% Find w_t and eps_t for hairdryer data

w_t = filter( foundModel.A, foundModel.C, x );   
w_t = w_t(length(foundModel.A):end );

eps_t = filter( foundModel.A, foundModel.C, y );   
eps_t = eps_t(length(foundModel.A):end );

%% CCF from w_t to eps_t

M = 40;
stem(-M:M,crosscorr(w_t, eps_t,M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% Find A2 and B

A2 = [1 0 1]; %since r = 2
B = [0 0 0 0 0 0]; %since d = 3, s = 2
Mi = idpoly([1], [B], [], [], [A2]);

Mi.Structure.b.Free = [0 0 0 1 1 1];

z = iddata(y,x, sampDist);
Mba2 = pem(z, Mi); 
present(Mba2);
etilde = resid(Mba2, z).y;
plotBasics(etilde); %etilde is not white noise

%% Calculate the residual etilde and verify it is uncorrelated with xt

stem(-M:M,crosscorr(etilde, x ,M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% Find C1 and A1

% The function call is estimateBJ( y, x, C1, A1, B, A2, titleStr, noLags )
C1 = [1];
A1 = [1 1] 

[foundModel, ey] = estimateBJ(y, x, C1, A1, Mba2.B, Mba2.F, 'BJ model 1', 20);

%% Crosscorrelation between x and etilde

etilde = y - filter( foundModel.B, foundModel.F, x );

etilde  = etilde(length(foundModel.B):end );
filter_xt = x(length(foundModel.B):end );

stem(-M:M,crosscorr(etilde, filter_xt ,M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% Estimate all polynomials again using pem for hairdryer data

A1 =[1 0];
A2 = [1 0 0];
B = [1 0 0 0 0 0];
C = 1;
Mi = idpoly(1,B,C,A1,A2);
z = iddata(y,x, sampDist);
MboxJ = pem(z,Mi);
present(MboxJ);
ehat = resid(MboxJ,z).y;

M = 40;
stem(-M:M,crosscorr(x,ehat, M));
title('Cross correlation function')
xlabel('Lags')
hold on
plot(-M:M,2/sqrt(n)*ones(1,2*M+1),'-')
plot(-M:M,-2/sqrt(n)*ones(1,2*M+1),'-')
hold off

%% 2.3 Prediction of ARMA-process

load 'svedala.mat'
y = svedala;
A = [1 -1.79 0.84];
C = [1 -0.18 -0.11];

k = 1;
[Fk, Gk] = polydiv(C, A, k);
yhat_k = filter(Gk,C,y);

yhat_k = yhat_k(length(Gk):end);
yk = y(length(Gk):end);

e_k = yk - yhat_k;
varNoise = var(e_k);

%% K = 3

k = 3;
[Fk, Gk] = polydiv(C, A, k);
yhat_k = filter(Gk,C,y);

yhat_k = yhat_k(length(Gk):end);
yk = y(length(Gk):end);

e_k = yk - yhat_k;
var_k3 = var(e_k);
mean_k3 = mean(e_k);

varTheo_k3 = Fk*Fk'*varNoise;

CI_k3 = [mean_k3 - norminv(0.975)*sqrt(var_k3) mean_k3 + norminv(0.975)*sqrt(var_k3)];

sum(abs(e_k)>CI_k3)

figure
subplot(311);
plot(covf(e_k,40))
subplot(312);
plot(e_k)

%% K = 26

load 'svedala.mat'
y = svedala;
A = [1 -1.79 0.84];
C = [1 -0.18 -0.11];

k = 26;
[Fk, Gk] = polydiv(C, A, k);
yhat_k = filter(Gk,C,y);

yhat_k = yhat_k(length(Gk):end);
yk = y(length(Gk):end);

e_k = yk - yhat_k;
var_k26 = var(e_k);
mean_k26 = mean(e_k);

varTheo_k26 = Fk*Fk'*varNoise;

CI_k26 = [mean_k26 - 2*sqrt(var_k26) mean_k26 + 2*sqrt(var_k26)];

sum(abs(e_k)>CI_k26)

%% Prediction of ARMAX-process

load 'svedala.mat'
load 'sturup.mat'

y = svedala;
x = sturup;

A = [1 -1.49 0.57];
B = [0 0 0 0.28 -0.26];
C = [1];

%% 26-step prediction

k = 26;
[Fk, Gk] = polydiv(C, A, k);
[Fhat, Ghat] = polydiv( conv(B,Fk), C, k );

%% Prediction for x

% plot(x);
plotBasics(x, 40);

A24 = [1 zeros(1, 23) -1];
x_s = myFilter(A24,1,x);
%plot(x_s);

foundModelX = estimateARMA(x_s, [1 1 1], [1 0 0 1 0 zeros(1,18) 1 1], 'Model for x', 40);
%foundModelX = estimateARMA(x_s, [1 1 1], [1 0 0 1 zeros(1,20) 1 1], 'Model for x', 40);
res = resid(foundModelX, x_s).y;
analyzets(res);

foundModelX.A = conv(A24, foundModelX.A); %Add the differentiation

ex = myFilter( foundModelX.A, foundModelX.C, x); 

% figure
% plot(ex);
checkIfWhite(ex);

%% Predict the input

k = 3;
[Fx, Gx] = polydiv(foundModelX.C, foundModelX.A, k);
xhat_k = filter(Gx,foundModelX.C, x);

modelLim = 100;

n = length(foundModelX.C) + k;
%xhat_k = xhat_k(n:end);
x_cut = x(n:end);

figure
plot(x_cut,'b')
hold on
plot(xhat_k,'r')

%% Take a look at predicition error for X
ehat = xhat_k - x;
ehat = ehat(100:end);

noLags = 40;

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );

%% Prediction for y

x_hat = filter(Ghat, C, x); 
y_hat = filter(Gk, C, y); 
x_pred_hat = filter(Fhat, 1, xhat_k); 

yhat_k = x_hat + y_hat + x_pred_hat;

%% Cut yhat_k

inSamp = [length(Fhat) length(Ghat) length(Gk) length(Gx)];
omitt = max(inSamp);

yhat_k = yhat_k(omitt:end);
y_cut = y(omitt:end);

shiftK = round( mean( grpdelay(Ghat, 1) ) );    % Compute the average group delay of the filter.
modelLim  = 90;

figure
plot(y_cut(100:300),'b')
hold on
plot(yhat_k(100:300),'r')

figure
plot([y_cut(1:end-shiftK) yhat_k(shiftK+1:end)] )
title(sprintf('Predicted output signal, y_{t+%i|t}', k) )

%% Prediction error of y

ehat = y_cut - yhat_k;
ehat = ehat(n:end);
noLags = 40;

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step output prediction residual', k) )
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );

%% Prediction of SARIMA-process

load 'svedala.mat'
y = svedala;

plot(y);
plotBasics(y,40);

%% Remove season

S = 24;
AS = [1 zeros(1, S-1) -1];
y_s = myFilter(AS, 1, y);
plotBasics(y_s, 40);

%% Find model for y

[foundModel, res] = estimateARMA(y_s, [1 1 1], [1 zeros(1, S-1) 1], 'Model for svedala', 40);

whitenessTest(res);

%% 
foundModel.A = conv(AS, foundModel.A);

ey = myFilter(foundModel.A, foundModel.C, y);
whitenessTest(ey);
plotBasics(ey);
var(ey); % = 0,3657

%% Predict future for k = 3

k = 3;
[Fhat, Ghat] = polydiv(foundModel.C, foundModel.A, k);
yhat_k = filter(Ghat, foundModel.C, y);

n = length(foundModel.C) + k;
yhat_k = yhat_k(n:end);
y_cut = y(n:end);
plot(y_cut,'b')
hold on
plot(yhat_k,'r')

%Error of prediction for k = 3

e_k3 = yhat_k - y_cut;
var_ek3 = var(e_k3);
plotBasics(e_k3);

%% Predict future for k = 26

k = 26;
[Fhat, Ghat] = polydiv(foundModel.C, foundModel.A, k);
yhat_k = filter(Ghat, foundModel.C, y);

n = length(foundModel.C) + k;
yhat_k = yhat_k(n:end);
y_cut = y(n:end);
plot(y_cut,'b')
hold on
plot(yhat_k,'r')

%Error of prediction for k = 26

e_k26 = yhat_k - y_cut;
var_ek26 = var(e_k26);
plotBasics(e_k26);







