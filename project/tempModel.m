%%% Create model for data temp

close all;
clc;

load('/Users/Hanna/Documents/MATLAB/Matlab_tsa_21/project/proj21data.mat')

plot(temp(1:1827));
tempData = temp(1:1096);
plot(tempData);
tempDataValidation = temp(1097:1461);
tempDataTest = temp(1462:1827);

%% Transform data?

bcNormPlot(tempData); %% No need for transform as lambda = 1

%% Remove outliers

tempData = filloutliers(tempData,'linear');

%% Remove trend 

tempDataZeroMean = tempData - mean(tempData);

A1 = [1 -1];
y = myFilter(A1, 1, tempDataZeroMean);
yZeroMean = y - mean(y);

plot(yZeroMean);
plotBasics(yZeroMean, 40);

%% Find ARMA-model

[foundModelTemp, resTemp] = estimateARMA(yZeroMean, [1 1], [1 1], 'Model temperature', 60);

whitenessTest(resTemp);

%% Check if normal

% What does the D'Agostino-Pearson's K2 test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'D' );
checkIfNormal( pacfEst(2:end), 'PACF', 'D' );

% What does the Jarque-Bera test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'J' );
checkIfNormal( pacfEst(2:end), 'PACF', 'J' );

%% Add differentiation to A-pol

foundModelTemp.A = conv(A1, foundModelTemp.A);

etilde = myFilter(foundModelTemp.A, foundModelTemp.C, tempDataZeroMean); 
whitenessTest(etilde);

%% Check ACF and PACF
[acfEst, pacfEst] = plotACFnPACF(etilde, 20, 'Model temperature' );     % Note that this function is now updated. Download it again!

% What does the D'Agostino-Pearson's K2 test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'D' );
checkIfNormal( pacfEst(2:end), 'PACF', 'D' );

%% Predict future to see if model fits with validation data

k = 1;
[Fhat, Ghat] = polydiv(foundModelTemp.C, foundModelTemp.A, k);
yhat_k = filter(Ghat, foundModelTemp.C, tempDataValidation);

n = length(Ghat);

yhat_k = yhat_k(n:end);
y_cut = tempDataValidation(n:end);

plot(y_cut,'b')
hold on
plot(yhat_k,'r')

%% Prediction error for validation data

e_hat_validation = y_cut - yhat_k;
noLags = 40;

figure
acfEst = acf( e_hat_validation, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat_validation );

%% What does the D'Agostino-Pearson's K2 test indicate?
% As the PACF should ring for an MA(k-1) process, we only check the ACF.
checkIfNormal( acfEst(k+1:end), 'ACF' );

%% Prediction residuals for the test data 

%Create prediction
k = 1;
[Fhat, Ghat] = polydiv(foundModelTemp.C, foundModelTemp.A, k);
yhat_k_test = filter(Ghat, foundModelTemp.C, tempDataTest);

n = length(Ghat) + k;

yhat_k_test = yhat_k_test(n:end);
y_cut_test = tempDataTest(n:end);

%Compare with real data
plot(y_cut_test,'b')
hold on
plot(yhat_k_test,'r')

%% Residual of test data

e_hat_test = y_cut_test - yhat_k_test;
noLags = 40;

figure
acfEst = acf( e_hat_test, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat_test );

% As the PACF should ring for an MA(k-1) process, we only check the ACF.
checkIfNormal( acfEst(k+1:end), 'ACF' );
