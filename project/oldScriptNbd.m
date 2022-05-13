%% Old script Nbdmodel

%%%%%%%%%%%%%%%%% OLD SCRIPT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add an offset

offset = 1;
nbdDataOffset = nbdDataNonNeg + offset;
plot(nbdDataOffset);

%% Transform the data

%lambda_max = bcNormPlot(nbdDataOffset);
lambda_max = 0;

nbdDataTransformed = log(nbdDataOffset);

%plot(nbdDataTransformed);
%plotBasics(nbdDataTransformed, 40);

nbdDataTransformedZeroMean = nbdDataTransformed - mean(nbdDataTransformed);
plot(nbdDataTransformedZeroMean);

%% Find ARMA-model

[foundModelNbd, resNbd] = estimateARMA(nbdDataTransformedZeroMean, [1 1 1], [1 1], 'Model Nbd', 40);

analyzets(resNbd); %% Why is not the residual normal distributed, or is it?

%% Check that ACF and PACF is normally distributed 
[acfEst, pacfEst] = plotACFnPACF(resNbd, 20, 'Model Nbd' );

% What does the D'Agostino-Pearson's K2 test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'D' );
checkIfNormal( pacfEst(2:end), 'PACF', 'D' );

% What does the Jarque-Bera test indicate?
checkIfNormal( acfEst(2:end), 'ACF', 'J' );
checkIfNormal( pacfEst(2:end), 'PACF', 'J' );

%% Transform validation data

nbdDataNonNegValidation = max(nbdData,0);
nbdDataOffsetValidation = nbdDataNonNegValidation + offset;
nbdDataTransformedValidation = log(nbdDataOffsetValidation);
nbdDataTransformedZeroMeanValidation = nbdDataTransformedValidation - mean(nbdDataTransformedValidation);

%% Predict future to see if model fits with the model data

k = 1;
[Fhat, Ghat] = polydiv(foundModelNbd.C, foundModelNbd.A, k);
yhat_k = filter(Ghat, foundModelNbd.C, nbdDataTransformedZeroMeanValidation);

n = length(Ghat);

yhat_k_transformed = exp(yhat_k);
yhat_k_transformed = yhat_k_transformed(n:end);

y_cut = nbdDataNonNegValidation(n:end);

plot(y_cut,'b')
hold on
plot(yhat_k_transformed,'r')

%% Prediction error

e_hat = y_cut - yhat_k_transformed;
noLags = 40;

figure
acfEst = acf( e_hat, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( e_hat );
