%% Part A - Model waterflow

data = waterflow(1:1827);
plot(data)
plotBasics(data, 60);
m = mean(data); 
dataZeroMean = data - m; % Make data zero-mean

%% Transform data with Box-cox

lambda = 0;
transdat = boxcox(lambda, data);
plot(transdat);

transdatZeroMean = transdat - mean(transdat);
plot(transdatZeroMean);
plotBasics(transdatZeroMean, 40);

%% Remove trend

trendpol = [1 -1];
y = myFilter(trendpol, 1, transdatZeroMean);
yZeroMean = y - mean(y);

plot(yZeroMean);
plotBasics(yZeroMean, 40);


%% Find ARMA-model

[foundModelA, resA] = estimateARMA(yZeroMean, [1 1 0 1 0 1], [1 1 1], 'ARMA-model', 40);
analyzets(resA); %% Why is not the residual normal distributed?

%% HH

foundModelA.A = conv(trendpol, foundModelA.A);

e_transdata = myFilter(foundModelA.A, foundModelA.C, transdat); 

analyzets(e_transdata);

%% Predict future

k = 3;
[Fhat, Ghat] = polydiv(foundModelA.C, foundModelA.A, k);
yhat_k = filter(Ghat, foundModelA.C, transdat);

n = length(foundModelA.C) + k;
yhat_k = yhat_k(n:end);
y_cut = transdat(n:end);
plot(y_cut,'b')
hold on
plot(yhat_k,'r')


