%% Part A - Model waterflow
%%  Split data into 2 and then into model, validation and test - sets

load('/Users/Hanna/Documents/MATLAB/Matlab_tsa_21/project/proj21data.mat')
waterflowData = waterflow(1:1096);
waterflowDataVal = waterflow(1097:1461);
waterflowDataTest = waterflow(1462:1827);
%% Visualise the different data sets
plot(waterflow(1:1827))
xline(1096, '-r')
hold on
xline(1461, '-r')
hold off
%% Plot model data 
plot(waterflowData);
xlabel('Days')
ylabel('Waterflow')
title('Data over the waterflow in the river at Emsfors')

%% Plot ACF n PACF and make the data zero mean
plotBasics(waterflowData, 20);
m = mean(waterflowData); 
dataZeroMean = waterflowData - m; % Make data zero-mean

%% Transform data with Box-cox
bcNormPlot(waterflowData);
title('Box Cox Normplot for model data')
%% Transform
lambda = 0;
transdat = boxcox(lambda, waterflowData);
plot(transdat);
xlabel('Days')
ylabel('Log(Waterflow)')
title('Log transformed model data over water flow')
%% Remove mean and plot ACF n PACF
transdatZeroMean = transdat - mean(transdat);

plot(transdatZeroMean);
plotBasics(transdatZeroMean, 20);

%% Remove trend
trendpol = [1 -1];
y = myFilter(trendpol, 1, transdatZeroMean);
yZeroMean = y - mean(y);

plot(yZeroMean);
title('Transformed and differentiated model data set')
checkIfWhite(yZeroMean);
plotBasics(yZeroMean, 20);

%% Find ARMA-model
[foundModelA, resA, acfEst, pacfEst] = estimateARMA(yZeroMean, [1 1 0 0 0 1], [1 1 1], 'Model A', 40);
%% Check if ACF and PACF is normal distributed
checkIfNormal( acfEst(2:end), 'ACF', 'D' );
checkIfNormal( pacfEst(2:end), 'PACF', 'D' ); % ACF n PACF normaldsitr

%% Add differentiation
foundModelA.A = conv(trendpol, foundModelA.A);
present(foundModelA);
noLags = 20;
e_transdata = myFilter(foundModelA.A, foundModelA.C, transdatZeroMean); 
%% Check ACF n PACF for residual
[acfEst, pacfEst] = plotACFnPACF( e_transdata, noLags, 'Model residual' );

checkIfNormal( acfEst(2:end), 'ACF', 'D' );
checkIfNormal( pacfEst(2:end), 'PACF', 'D' ); % ACF n PACF normaldsitr
checkIfWhite(e_transdata);
%% Predict future to see if model fits model data

k = 7;
[Fhat, Ghat] = polydiv(foundModelA.C, foundModelA.A, k);
yhat_k = filter(Ghat, foundModelA.C, transdat);

n = length(Ghat);

yhat_k = yhat_k(n:end);
yhat_k_transformed = exp(yhat_k);

y_cut = waterflowData(n:end);
%% Plot prediction for modeling data
plot(y_cut,'b')
hold on
plot(yhat_k_transformed,'r')

%% Prediction error
ehat = y_cut - yhat_k_transformed;
noLags = 40;

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)

% pacfEst = pacf( ehat, noLags, 0.05 );
% checkIfNormal( pacfEst(k+1:end), 'PACF' );
var(ehat)

%% Transform validation data
transVal = log(waterflowDataVal);
%% Predict validation data
k = 1;
[Fhat, Ghat] = polydiv(foundModelA.C, foundModelA.A, k);
yhat_k = filter(Ghat, foundModelA.C, transVal);

n = length(Ghat);

yhat_k = yhat_k(n:end);
yhat_k_transformed = exp(yhat_k);
%yhat_k_transformed = yhat_k_transformed(end-30:end);
y_cut = waterflowDataVal(n:end);

%y_cut_month = waterflowDataVal(end-30:end);
%% Plot prediction for validation data
plot(y_cut,'b')
hold on
plot(yhat_k_transformed,'r')
xlabel('Days')
ylabel('Waterflow')
title('One Step Prediction')
legend('Output signal', 'Predicted output')
%% Shift
shiftK = round( mean( grpdelay(Ghat, 1) ) );
pstart= 0;
f1 = figure;
plot([y_cut(1:end-shiftK) yhat_k_transformed(shiftK+1:end)] )
figProp = get(f1);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
title( sprintf('Shifted %i-step predictions',k))
xlabel('Days')
ylabel('Waterflow')
legend('Measured data', 'Predicted data','Location','NE')
%% Prediction error for validation data
ehat = y_cut - yhat_k_transformed;
noLags = 20;
%% Plot ACF for residual
figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) );
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1);
%% Check if normal and 
checkIfWhite(ehat);
acfEst = acf( ehat, noLags, 0.05 );
checkIfNormal( acfEst(k+1:end), 'ACF' );

%% Variance of prediction error for validation data
var(ehat)
%% 7-step prediction for the last month of the val-data
k = 7;
[Fhat, Ghat] = polydiv(foundModelA.C, foundModelA.A, k);
yhat_k = filter(Ghat, foundModelA.C, transVal);

n = length(Ghat);

yhat_kLastMonth = yhat_k(end-37:end);
yhat_k_transformedLastMonth = exp(yhat_kLastMonth);
y_LastMonth = waterflowDataVal(end-37:end);

%% Plot 7 step prediction for last month
plot(yhat_k_transformedLastMonth)
hold on
plot(y_LastMonth)
hold off

%% Shift
shiftK = round( mean( grpdelay(Ghat, 1) ) );
pstart= 0;
f1 = figure;
plot([y_LastMonth(1:end-shiftK) yhat_k_transformedLastMonth(shiftK+1:end)] )
figProp = get(f1);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
title( sprintf('Shifted %i-step predictions',k))
xlabel('Days')
ylabel('Waterflow')
legend('Measured data', 'Predicted data','Location','NE')

%% Predict on test data
transTest = log(waterflowDataTest);
%% Predict on test data
k = 1;
[Fhat, Ghat] = polydiv(foundModelA.C, foundModelA.A, k);
yhat_k = filter(Ghat, foundModelA.C, transTest);

n = length(Ghat);

yhat_k = yhat_k(n:end);
yhat_k_transformed = exp(yhat_k);
y_cut = waterflowDataTest(n:end);
%% Plot prediction for test data
plot(y_cut,'b')
hold on
plot(yhat_k_transformed,'r')

%% Shift
shiftK = round( mean( grpdelay(Ghat, 1) ) );
pstart= 0;
f1 = figure;
plot([y_cut(1:end-shiftK) yhat_k_transformed(shiftK+1:end)] )
figProp = get(f1);
line( [pstart pstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
title( sprintf('Shifted %i-step predictions',k))
xlabel('Days')
ylabel('Waterflow')
legend('Measured data', 'Predicted data','Location','NE')

%% Prediction error for test data
ehat = y_cut - yhat_k_transformed;
noLags = 20;

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) );
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1);

checkIfWhite(ehat);
acfEst = acf( ehat, noLags, 0.05 );
checkIfNormal( acfEst(k+1:end), 'ACF' );
%% Variance of prediction error of test data
var(ehat)