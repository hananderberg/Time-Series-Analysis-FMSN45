%% Model for external signal input 

load('/Users/Hanna/Documents/MATLAB/Matlab_tsa_21/project/proj21data.mat')
q = 1;
p = 120;
modelLim = 1096;

x1 = nbd(q:1096);
x2 = temp(q:1096);
y = waterflow(q:1096);

x1 = x1(p:end);
x2 = x2(p:end);
y = y(p:end);

figure
plot(x1)
hold on
plot(y)
plot(x2);

%Remove -1 and replace with middle value
x1(x1 == -1) = NaN;
x1 = fillmissing(x1,'linear');

x2 = filloutliers(x2,'linear');

% Fix validation
x1_validation = nbd(1097:1461);
x1_validation(x1_validation == -1) = NaN;
x1_validation = fillmissing(x1_validation,'linear');

x2_validation = temp(1097:1461);
x2_validation = filloutliers(x2_validation,'linear');

y_validation = waterflow(1097:1461);

x1_test = nbd(1462:1827);
x1_test(x1_test == -1) = NaN;
x1_test = fillmissing(x1_test,'linear');

x2_test = temp(1462:1827);
y_test = waterflow(1462:1827);

% Make zero-mean
y = y - mean(y);
x1 = x1 - mean(x1);
x2 = x2 - mean(x2);

x1_validation = x1_validation - mean(x1_validation);
x2_validation = x2_validation - mean(x2_validation);
y_validation = y_validation - mean(y_validation);

x1_test = x1_test - mean(x1_test);
x2_test = x2_test - mean(x2_test);
y_test = y_test - mean(y_test);

%% Look at crosscorrelation y, x1 and x2

CCF(y, x1, 60);
title('Crosscorrelation between waterflow and rain')
CCF(y, x2, 60);
title('Crosscorrelation between temperature and waterflow')

%% Find w_t and eps_t for tempData and waterflowData

w_t = filter(foundModelTemp.A, foundModelTemp.C, x2);   
w_t = w_t(length(foundModelTemp.A):end);
whitenessTest(w_t); % w_t is white! 

eps_t = filter(foundModelTemp.A, foundModelTemp.C, y);   
eps_t = eps_t(length(foundModelTemp.A):end);

figure
plot(w_t, 'b');
hold on
plot(eps_t, 'r');

%% CCF from w_t to eps_t

CCF(eps_t, w_t, 50);

%% Find A2 and B

%B(z) = B(z),    F(z) = A2(z)

A2 = [1]; %since r = 2
B = 0.3*[1]; %since d = ?, s = 2
Mi = idpoly([1],[B],[],[],[A2]);

%A22(2:end) = A22(2:end) * 0,3;

Mi.Structure.B.Free = [1];
%Mi.Structure.F.Free = [0 1 1 1 1 1];

z = iddata(y,x2);
Mba2 = pem(z,Mi);
present(Mba2);

e_tilde = resid(Mba2,z).y; 

% Calculate the residual etilde and verify it is uncorrelated with x2
e_tilde = e_tilde(length(Mba2.B):end);
filter_xt = x2(length(Mba2.B):end);

CCF(filter_xt, e_tilde, 100);

figure
zplane(Mba2.B);
figure
zplane(Mba2.F);

%% Compare etilde and y

figure
plot(e_tilde, 'b')
hold on
plot(y, 'g');

%% Next input - raindata

z = iddata(y, x2);
y2 = resid(Mba2, z).y; y2 = y2(length(Mba2.A):end );
y2 = y - filter(Mba2.B, Mba2.F, x2); y2 = y2(length(Mba2.A):end );

plotBasics(y2, 40);
title('Residual from just temperature as input');

%% Find w_t2 and eps_t2

w_t2 = filter(foundModelNbd.A, foundModelNbd.C, x1);   
w_t2 = w_t2(length(foundModelNbd.A):end);
whitenessTest(w_t2); %w_t2 is white! 

eps_t2 = filter(foundModelNbd.A, foundModelNbd.C, y2);   
eps_t2 = eps_t2(length(foundModelNbd.A):end);

figure
plot(w_t2, 'b');
hold on
plot(eps_t2, 'r');

%% Look at crosscorrelation

CCF(eps_t2, w_t2, 100);

%% Estimate BJ-model

% A22 = [1 1]; 
% B2 = [1 1 1];
% A1 = [1 1 1];
% C1 = [1 0 1];

% A22 = [1];
% B2 = [zeros(1,9) 1];
% A1 = [1 1 1];
% C1 = [1 1];

A22 = [1 1 1];
B2 = [1];
A1 = [1 1 1 1 0 1];
C1 = [1 1 1 1];

[foundModel, res, acfEst, pacfEst] = estimateBJ(y2, x2, [C1], [A1], [B2], [A22], 'BJ model 1', noLags );

%% Find A2 and B for y2 and x1

%B(z) = B(z),    F(z) = A2(z)

% A22 = [1 1]; %since r = 1
% B2 = [zeros(1,9) 1]; %since d = 10, s = 2 

A22 = [1]; %since r = 1
B2 = 0.3*[1 1 1]; %since d = 10, s = 2 

A22(2:end) = A22(2:end) * 0,3;
B2(2:end) = B2(2:end) * 0,3;

Mi2 = idpoly([1],[B2],[],[],[A22]);

%Mi2.Structure.F.Free = [0 1 1];
Mi2.Structure.B.Free = [1 1 1];

z = iddata(y2,x1);
Mba3 = pem(z,Mi2);
present(Mba3);

e_tilde_2 = resid(Mba3,z).y; % Residual from y when removing x2 and x1, should be y, not white noise

e_tilde_2 = e_tilde_2(length(Mba3.B):end);
filter_xt_2 = x1(length(Mba3.B):end);

CCF(filter_xt_2, e_tilde_2, 100);

figure
zplane(Mba3.B);
figure
zplane(Mba3.F);

%% Compare etilde_2 and y2

figure
plot(e_tilde_2, 'b')
hold on
plot(y2, 'g');

%% Form a BJ-model

% A1 = [1 1 1];
% C1 = [1 1];

A1 = [1 1 1];
C1 = [1];

A1(2:end) = A1(2:end) * 0,3;
C1(2:end) = C1(2:end) * 0,3;

Mi3 = idpoly([A1],[],[C1]);

Mi3.Structure.A.Free = [0 1 1];
%Mi3.Structure.C.Free = [0 0 1];

e_tilde_2_data = iddata(e_tilde_2);
Mba4 = pem(e_tilde_2_data, Mi3);
present(Mba4);
e_res = resid(Mba4,e_tilde_2_data).y;

plotBasics(e_res, 20);
whitenessTest(e_res);

% Check roots in zplane
figure
zplane(Mba4.A);
figure
zplane(Mba4.C);

% Compare e_res and y

figure
plot(e_res, 'b')
hold on
plot(y, 'g');

%% Check if pacf and acf is normally distributed

[pacfEst] = pacf(e_res, 100);
[acfEst] = acf(e_res, 100);
checkIfNormal( acfEst(2:end), 'ACF' );
checkIfNormal( pacfEst(2:end), 'PACF' );

%% Re-estimate all parameters using pem

% This means that:
%   A(z) = 1,       B(z) = B(z),    F(z) = A2(z)
%   C(z) = C1(z),   D(z) = A1(z)

A = 1;
B = { Mba2.B , Mba3.B };
F = { Mba2.F , Mba3.F };
C = Mba4.C;
D = Mba4.A;
modelInput = idpoly(A,B,C,D,F); %initial values for all polynomials

% modelInput.Structure.B(1,1).Free = [0 0 1 1];
% modelInput.Structure.B(1,2).Free = [1 1 0];
% modelInput.Structure.C.Free = [0 1 1];

modelOutput = setPolyFormat(modelInput,'double'); %set format to two inputs

z_final = iddata(y,[x2,x1]); %both inputs
MboxJ = pem(z_final, modelOutput); %estimate all parameters 
present(MboxJ); %present model
res = resid(z_final, MboxJ).y;

plotBasics(res);
figure
whitenessTest(res)

%% Check if pacf and acf is normally distributed

[pacfEst] = pacf(res, 100);
[acfEst] = acf(res, 100);
checkIfNormal( acfEst(2:end), 'ACF' );
checkIfNormal( pacfEst(2:end), 'PACF' );

%% Check if there is any correlation left

c = 1;

CCF(x1(c:end), res(c:end))
CCF(x2(c:end), res(c:end))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Lets predict the input first.

allTempData = temp(1:1096);
allTempData = allTempData - mean(allTempData);

% Replace all -1 with mean values
allRainData = nbd(1:1096);
allRainData(allRainData == -1) = NaN;
allRainData = fillmissing(allRainData,'linear');
allRainData = allRainData - mean(allRainData);

k = 1;

%Predict temperature
[F_temp, G_temp] = polydiv( foundModelTemp.C, foundModelTemp.A, k );
xhatk_temp = filter(G_temp, foundModelTemp.C, allTempData);

%Predict rain
[F_rain, G_rain] = polydiv( foundModelNbd.C, foundModelNbd.A, k );
xhatk_rain = filter(G_rain, foundModelTemp.C, allRainData);

n = length(G_temp)+k;

xhatk_temp = xhatk_temp(n:end);
xhatk_rain = xhatk_rain(n:end);
allRainData_cut = allRainData(n:end);
allTempData_cut = allTempData(n:end);

N = length(allRainData);

% Compute the average group delay.
shiftK = round( mean( grpdelay(G_temp, 1) ) );

figure
plot([allTempData_cut(1:end-shiftK) xhatk_temp(shiftK+1:end)] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
legend('Input signal', 'Predicted input', 'Prediction starts')
title( sprintf('Shifted predicted input signal, x_{t+%i|t}', k) )
axis([1 N min(x2)*1.5 max(x2)*1.5])

figure
plot([allRainData_cut(1:end-shiftK) xhatk_rain(shiftK+1:end)] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
legend('Input signal', 'Predicted input', 'Prediction starts')
title( sprintf('Shifted predicted input signal, x_{t+%i|t}', k) )
axis([1 N min(x2)*1.5 max(x2)*1.5])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Proceed to predict the data using the predicted input.
% Form the BJ prediction polynomials. In our notation, these are
%   A1 = foundModel.D
%   C1 = foundModel.C
%   A2 = foundModel.F
% 
% The KA, KB, and KC polynomials are formed as:
%   KA = conv( A1, A2 );
%   KB = conv( A1, B );
%   KC = conv( A2, C1 );

% A1 = MboxJ.D;
% C1 = MboxJ.C;
% B1 = cell2mat(MboxJ.B(1));
% B2 = cell2mat(MboxJ.B(2));
% A21 = cell2mat(MboxJ.F(1));
% A22 = cell2mat(MboxJ.F(2));

A1 = MboxJ.D;
C1 = MboxJ.C;
B1 = MboxJ.B(1);
B2 = MboxJ.B(2);
A21 = MboxJ.F(1);
A22 = MboxJ.F(2);

KA = conv( A1, A21 ); KA = conv( KA, A22 );
KB1 = conv( A1, A22 ); KB1 = conv( KB1, B1); %for temperature
KB2 = conv( A1, A21 ); KB2 = conv( KB2, B2); %for rain
KC = conv( A21, A22 ); KC = conv( KC, C1 );

%% Predict using model data

k = 1;
modelLim = 1;
N = length(y_test);

% Predict temperature x2
[Fx, Gx] = polydiv( foundModelTemp.C, foundModelTemp.A, k );
xhatk_temp = filter(Gx, foundModelTemp.C, x2 );

%Predict rain x1
[Fx, Gx] = polydiv( foundModelNbd.C, foundModelNbd.A, k );
xhatk_rain = filter(Gx, foundModelNbd.C, x1);

%Prediction with 2 inputs

% Form the ARMA prediction for y_t (note that this is not the same G
% polynomial as we computed above (that was for x_t, this is for y_t).
[Fy, Gy] = polydiv( C1 , A1, k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials for temperature
[Fhh_temp, Ghh_temp] = polydiv( conv(Fy, KB1), KC, k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials.
[Fhh_rain, Ghh_rain] = polydiv( conv(Fy, KB2), KC, k );

% Form the predicted output signal using the predicted input signal.
tempPred = filter(Fhh_temp, 1, xhatk_temp);
temp = filter(Ghh_temp, KC, x2); 
rainPred = filter(Fhh_rain, 1, xhatk_rain); 
rain = filter(Ghh_rain, KC, x1);
yPred = filter(Gy, KC, y);

yhatk = tempPred + temp + rainPred + rain + yPred;

N = length(y);
pstart = 0;

f2 = figure;
shiftK_y = round( mean( grpdelay(Fhh_rain, 1) ) );
plot([y(1:end-shiftK_y) yhatk(shiftK_y+1:end)] )
figProp = get(f2);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
xlim([0 N])
title( sprintf('Shifted %i-step predictions of y(t)',k))
legend('y(t)', 'Predicted data', 'Prediction starts','Location','NW')

%% Prediction error for model data
ehat = y - yhatk;
ehat = ehat(20:end);
var(ehat)

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step output prediction residual', k) )
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict using test data

k = 1;
modelLim = 1;
N = length(y_test);

% Predict temperature x2
[Fx, Gx] = polydiv( foundModelTemp.C, foundModelTemp.A, k );
xhatk_temp = filter(Gx, foundModelTemp.C, x2_test  );

%Predict rain x1
[Fx, Gx] = polydiv( foundModelNbd.C, foundModelNbd.A, k );
xhatk_rain = filter(Gx, foundModelNbd.C, x1_test);

%Prediction with 2 inputs

% Form the ARMA prediction for y_t (note that this is not the same G
% polynomial as we computed above (that was for x_t, this is for y_t).
[Fy, Gy] = polydiv( C1 , A1, k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials for temperature
[Fhh_temp, Ghh_temp] = polydiv( conv(Fy, KB1), KC, k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials.
[Fhh_rain, Ghh_rain] = polydiv( conv(Fy, KB2), KC, k );

% Form the predicted output signal using the predicted input signal.
tempPred = filter(Fhh_temp, 1, xhatk_temp);
temp = filter(Ghh_temp, KC, x2_test); 
rainPred = filter(Fhh_rain, 1, xhatk_rain); 
rain = filter(Ghh_rain, KC, x1_test);
yPred = filter(Gy, KC, y_test);

yhatk = tempPred + temp + rainPred + rain + yPred;

N = length(y_test);
pstart = 0;

f2 = figure;
shiftK_y = round( mean( grpdelay(Fhh_rain, 1) ) );
plot([y_test(1:end-shiftK_y) yhatk(shiftK_y+1:end)] )
figProp = get(f2);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
xlim([0 N])
title( sprintf('Shifted %i-step predictions of y(t)',k))
legend('y(t)', 'Predicted data', 'Prediction starts','Location','NW')

%% Prediction error for test data
ehat = y_test - yhatk;
ehat = ehat(20:end);
var(ehat)

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step output prediction residual', k) )
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict using validation data

k = 1;
modelLim = 1;
N = length(y_validation);

% Predict temperature x2
[Fx, Gx] = polydiv( foundModelTemp.C, foundModelTemp.A, k );
xhatk_temp = filter(Gx, foundModelTemp.C, x2_validation  );

%Predict rain x1
[Fx, Gx] = polydiv( foundModelNbd.C, foundModelNbd.A, k );
xhatk_rain = filter(Gx, foundModelNbd.C, x1_validation);

%Prediction with 2 inputs

% Form the ARMA prediction for y_t (note that this is not the same G
% polynomial as we computed above (that was for x_t, this is for y_t).
[Fy, Gy] = polydiv( C1 , A1, k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials for temperature
[Fhh_temp, Ghh_temp] = polydiv( conv(Fy, KB1), KC, k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials.
[Fhh_rain, Ghh_rain] = polydiv( conv(Fy, KB2), KC, k );

% Form the predicted output signal using the predicted input signal.
tempPred = filter(Fhh_temp, 1, xhatk_temp);
temp = filter(Ghh_temp, KC, x2_validation); 
rainPred = filter(Fhh_rain, 1, xhatk_rain); 
rain = filter(Ghh_rain, KC, x1_validation);
yPred = filter(Gy, KC, y_validation);

yhatk = tempPred + temp + rainPred + rain + yPred;

N = length(y_validation);
pstart = 0;

f2 = figure;
shiftK_y = round( mean( grpdelay(Fhh_rain, 1) ) );
plot([y_validation(1:end-shiftK_y) yhatk(shiftK_y+1:end)] )
figProp = get(f2);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
xlim([0 N])
title( sprintf('Shifted %i-step predictions of y(t)',k))
legend('y(t)', 'Predicted data', 'Prediction starts','Location','NW')

%% Prediction error for validation data

ehat = y_validation - yhatk;
ehat = ehat(20:end);
var(ehat)

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step output prediction residual', k) )
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );
