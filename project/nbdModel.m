%% Create model for data ndb

close all;
clc;

load('/Users/Hanna/Documents/MATLAB/Matlab_tsa_21/project/proj21data.mat')

p = 1;

plot(nbd(1:1827));
nbdData = nbd(p:1096);
nbdDataValidation = nbd(1097:1461);
nbdDataTest = nbd(1462:1827);

%% Remove -1 and replace with mean

nbdData(nbdData == -1) = NaN;
nbdDataNonNeg = fillmissing(nbdData,'linear');
m = mean(nbdDataNonNeg);
nbdDataNonNeg = nbdDataNonNeg - m;

figure
plot(nbdDataNonNeg, 'b');
hold on
%plot(nbdData, 'r');

%% Should we remove any outliers?

figure
tacf(nbdDataNonNeg, 40, 0.04, 0.05, 'plotIt');

% Look at the acf
plotBasics(nbdDataNonNeg, 40); % Not a big difference between acf and tacf, no need to remove outliers. 

%% Find model

[foundModelNbd, resNbd, acfEst, pacfEst] = estimateARMA(nbdDataNonNeg, [1 1 1 1], [1 1], 'Model Nbd', 40);

%% Check if normal

% What does the D'Agostino-Pearson's K2 test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'D' );
checkIfNormal( pacfEst(2:end), 'PACF', 'D' );

% What does the Jarque-Bera test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'J' );
checkIfNormal( pacfEst(2:end), 'PACF', 'J' );

%% Predict future to see if model fits with the model data

k = 1;
[Fhat, Ghat] = polydiv(foundModelNbd.C, foundModelNbd.A, k);
yhat_k = filter(Ghat, foundModelNbd.C, nbdDataNonNeg);

n = length(Ghat)+k;

yhat_k = yhat_k(n:end) ;
y_cut = nbdDataNonNeg(n:end) + m;

plot(y_cut,'b')
hold on
plot(yhat_k,'r')

%% Prediction error

e_hat = y_cut - yhat_k;
noLags = 40;

figure
acfEst = acf( e_hat, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat );

var2 = var(e_hat);

%% Predict future to see if model fits with the validation data

k = 1;
[Fhat, Ghat] = polydiv(foundModelNbd.C, foundModelNbd.A, k);
yhat_k = filter(Ghat, foundModelNbd.C, nbdDataValidation);

n = length(Ghat)+k;

yhat_k = yhat_k(n:end);
y_cut = nbdDataValidation(n:end);

plot(y_cut,'b')
hold on
plot(yhat_k,'r')

%% Prediction error for validation data

e_hat = y_cut - yhat_k;
noLags = 40;

figure
acfEst = acf( e_hat, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat );

%% Predict future to see if model fits with the test data

k = 1;
[Fhat, Ghat] = polydiv(foundModelNbd.C, foundModelNbd.A, k);
yhat_k = filter(Ghat, foundModelNbd.C, nbdDataTest);

n = length(Ghat)+k;

yhat_k = yhat_k(n:end);
y_cut = nbdDataTest(n:end);

plot(y_cut,'b')
hold on
plot(yhat_k,'r')

%% Prediction error for test data

e_hat = y_cut - yhat_k;
noLags = 40;

figure
acfEst = acf( e_hat, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat );


