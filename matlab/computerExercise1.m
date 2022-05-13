%% 2.1

A1 = [ 1 -1.79 0.84 ];
C1 = [ 1 -0.18 -0.11 ];

A2 = [ 1 -1.79 ];
C2 = [ 1 -0.18 -0.11 ];

ARMA1_poly = idpoly(A1,[],C1);
ARMA2_poly = idpoly(A2,[],C2);

figure()
subplot(211);
pzmap(ARMA1_poly);
subplot(212);
pzmap(ARMA2_poly);

N = 600;
sigma2 = 1;
omitted = 100;

y1 = simulateMyARMA(ARMA1_poly.c, ARMA1_poly.a, N, sigma2, omitted);
y2 = simulateMyARMA(ARMA2_poly.c, ARMA2_poly.a, N, sigma2, omitted);

%% Plot y1 and y2

figure()
subplot(211);
plot(y1);
subplot(212);
plot(y2);

%% Estimate covariance 

sigma2 = 1.5;
m = 20;
r_theo = kovarians(ARMA1_poly.c, ARMA1_poly.a, m);
stem(0:m, r_theo * sigma2);
hold on
r_est = covf(y1, m+1);
stem(0:m, r_est, 'r');

%% MQ1

N1 = 650;
m1 = 20;
rng(0);
whiteNoiseProcess = randn(N1,1);
acf_est = acf(whiteNoiseProcess, m1);
plot(acf_est);


%% Plot acf, pacf, normplot

plotBasics(y1);


%% Re-estimate model

N = 300;
omitted = 100;
sigma2 = 1.5;
rng(0);
y1 = simulateMyARMA(ARMA1_poly.c,ARMA1_poly.a, N, sigma2, omitted);

data = iddata(y1);
na = 2;
nc = 2;

ar_model = arx(y1, na);
arma_model = armax(y1, [na nc]);
present(arma_model)

e_hat = filter(ARMA1_poly.c, ARMA1_poly.a, y1);
e_hat = e_hat(length(ARMA1_poly.a):end);

%% Adding one parameter at a time

rng(0);
na = 4;
nc = 1;

arma_model = armax(y1, [na nc]);
present(arma_model);
e_hat = myFilter(arma_model.a, arma_model.c, y1);
plotBasics(e_hat);

%% 2.2

load 'data.dat'
data = iddata(data);

load 'noise.dat'

ar1_model = arx(y1, 1);
ar2_model = arx(y1, 2);
ar3_model = arx(y1, 3);
ar4_model = arx(y1, 4);
ar5_model = arx(y1, 5);

rar1 = resid(ar1_model, data);
rar2 = resid(ar2_model, data);

plot(rar1.y);
hold on
plot(noise);

present(ar5_model);

%% ARMAX-modelling

am11_model = armax(data, [ 1 1 ]);
am12_model = armax(data, [ 1 2 ]);
am21_model = armax(data, [ 2 1 ]);
am22_model = armax(data, [ 2 2 ]);

resid11 = resid(am11_model, data);
%plotBasics(resid11.y);
resid12 = resid(am12_model, data);
%plotBasics(resid12.y);
resid21 = resid(am21_model, data);
%plotBasics(resid21.y);
resid22 = resid(am22_model, data);
%plotBasics(resid22.y);

present(am21_model) %Choose am21_model as MSE=1.146

%% 2.3 Estimation of SARIMA-process

rng(0);
N = 10000;
A = [1 -1.5 0.7];
C = [1 zeros(1,11) -0.5];
A12 = [1 zeros(1,11) -1];
A_star = conv(A, A12);
e = randn(N,1);
y = filter(C,A_star,e);
y = y(101:end);
plot(y);
plotBasics(y);

%% Remove the season 

y_s = filter(A12,1,y);
y_s = y_s(length(A12):end);
data = iddata(y_s);
%plot(data.y)

model_init = idpoly([ 1 0 0 ] , [ ] , [ 1 zeros(1,12)]);
model_init.Structure.c.Free = [zeros(1,12) 1];
model_armax = pem(data, model_init)

present(model_armax)

res = resid(model_armax, data).y;
%plot(res);
plotBasics(res);
whitenessTest(res); %Why do we use the upper bound for a*star1?

%% Estimation on real data

load('svedala.mat');
plot(svedala);
m = mean(svedala);
svedalaZeroMean = svedala - m;

plotBasics(svedala, 25);
plotBasics(svedalaZeroMean, 25);

%% Try season and remove it 

plotBasics(svedalaZeroMean, 25);
A24 = [1 zeros(1,23) -1];
svedalaZeroMean_s = myFilter(A24, 1, svedalaZeroMean); %% "starkt" sätt att ta bort trenden

plotBasics(svedalaZeroMean_s, 30);

%% Try AR(2)

data = iddata(svedalaZeroMean_s);
poly = idpoly([1 0 0],[],[]);
poly.Structure.a.Free = [0 1 1];
ar2_model = pem(data,poly);

ys = myFilter(ar2_model.c,ar2_model.a,data.y);
plotBasics(ys, 30);

res = resid(ar2_model, data).y;
plotBasics(res, 30);
whitenessTest(res);

%% Try ARMA(2,24)

newpoly = idpoly([1 0 0],[],[1 zeros(1,23) 1]);
newpoly.Structure.a.Free = [0 1 1];
newpoly.Structure.c.Free = [0 zeros(1,23) 1];

newmodel = pem(data, newpoly);

ys = myFilter(newmodel.c,newmodel.a,data.y);
plotBasics(ys, 30);

res = resid(newmodel, data).y;
plotBasics(res);
whitenessTest(res);

%% Try with function estimateARMA

estimateARMA(data, [1 0 0], [1 zeros(1,23) 1], 'Modeltest', 30);




