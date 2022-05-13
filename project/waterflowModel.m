%% Part A - Model waterflow

load('/Users/Hanna/Documents/MATLAB/Matlab_tsa_21/project/proj21data.mat')

%% Divide data into validation, test and model

plot(waterflow(1:1827));
waterflowData = waterflow(1:1096);
waterflowDataValidation = waterflow(1097:1461);
waterflowDataTest = waterflow(1462:1827);

plot(waterflowData);

%% Plot model data

plot(waterflowData);
plotBasics(waterflowData, 60);
m = mean(waterflowData); 
dataZeroMean = waterflowData - m; % Make data zero-mean

%% Find lambda

bcNormPlot(dataZeroMean); %% lambda = 0

%% Transform data with Box-cox

lambda = 0;
transdat = boxcox(lambda, waterflowData);
plot(transdat);
plotBasics(transdat, 40);

transdatZeroMean = transdat - mean(transdat);

figure 
plot(transdatZeroMean);
plotBasics(transdatZeroMean, 40);

%% Remove trend

trendpol = [1 -1];
y = myFilter(trendpol, 1, transdatZeroMean);
yZeroMean = y - mean(y);

plot(yZeroMean);
plotBasics(yZeroMean, 40);

%% Find ARMA-model

[foundModelA, resA] = estimateARMA(yZeroMean, [1 1 0 0 0 1], [1 1 1], 'Model A', 40);
%[foundModelA, resA] = estimateARMA(yZeroMean, [1 1 1 1 1 1], [1 1], 'Model A', 40);

%analyzets(resA); 
whitenessTest(resA)


%% Check ACF and PACF
[acfEst, pacfEst] = plotACFnPACF(yZeroMean, 20, 'Model A' );     % Note that this function is now updated. Download it again!

% What does the D'Agostino-Pearson's K2 test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'D' );
checkIfNormal( pacfEst(2:end), 'PACF', 'D' );

%% Add differentiation to A-pol

foundModelA.A = conv(trendpol, foundModelA.A);

e_transdata = myFilter(foundModelA.A, foundModelA.C, transdatZeroMean); 

present(foundModelA);
analyzets(e_transdata);
whitenessTest(e_transdata);
plotBasics(e_transdata);

%% Transform validation data

transdatValidation = log(waterflowDataValidation);

%% Predict future to see if model fits with validation data

k = 1;
[Fhat, Ghat] = polydiv(foundModelA.C, foundModelA.A, k);
yhat_k = filter(Ghat, foundModelA.C, transdatValidation);

n = length(Ghat);

yhat_k_transformed = exp(yhat_k);
yhat_k_transformed = yhat_k_transformed(n:end);

y_cut = waterflowDataValidation(n:end);
 
f2 = figure;
shiftK_y = round( mean( grpdelay(Ghat, 1) ) );
plot([y_cut(1:end-shiftK_y) yhat_k_transformed(shiftK_y+1:end)] )
figProp = get(f2);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
xlim([0 N])
title( sprintf('Shifted %i-step predictions of y(t)',k))
legend('y(t)', 'Predicted data', 'Prediction starts','Location','NW')

%% Prediction error

e_hat = y_cut - yhat_k_transformed;
%e_hat = e_hat(20:end);
noLags = 20;

figure
acfEst = acf( e_hat, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat );

%% What does the D'Agostino-Pearson's K2 test indicate?
% As the PACF should ring for an MA(k-1) process, we only check the ACF.
checkIfNormal( acfEst(k+1:end), 'ACF' );

%% Prediction residuals for the test data 

%Transform the test data
transdatTest = boxcox(lambda, waterflowDataTest);

%Create prediction
k = 7;
[Fhat, Ghat] = polydiv(foundModelA.C, foundModelA.A, k);
yhat_k_test = filter(Ghat, foundModelA.C, transdatTest);

n = length(Ghat) + k;

yhat_k_transformed_test = exp(yhat_k_test);
yhat_k_transformed_test = yhat_k_transformed_test(n:end);
y_cut_test = waterflowDataTest(n:end);

f2 = figure;
shiftK_y = round( mean( grpdelay(Ghat, 1) ) );
plot([y_cut_test(1:end-shiftK_y) yhat_k_transformed_test(shiftK_y+1:end)] )
figProp = get(f2);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
xlim([0 N])
title( sprintf('Shifted %i-step predictions of y(t)',k))
legend('y(t)', 'Predicted data', 'Prediction starts','Location','NW')

%% Residual of test data

e_hat_test = y_cut_test - yhat_k_transformed_test;
e_hat_test = e_hat_test(20:end);
noLags = 20;

figure
acfEst = acf( e_hat_test, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat_test );

% As the PACF should ring for an MA(k-1) process, we only check the ACF.
checkIfNormal( acfEst(k+1:end), 'ACF' );

%% Naive predictor

N = length(waterflowDataTest);

nav_pred = filter([1 -1],1,waterflow(1462:1827));

ehat_naive = waterflowDataTest - nav_pred;
ehat_naive = ehat_naive(20:end);
var_naive = var(ehat_naive);

figure
plot(nav_pred);
hold on
plot(waterflowDataTest);

figure
plot(ehat_naive);
hold on
plot(e_hat_test);
title( sprintf('The residuals of the naive predictor and the original predictor'))

