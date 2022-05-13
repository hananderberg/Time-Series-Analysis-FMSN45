%% Q1
N = 10000;
extraN = 100;
A = [1 -1.8 0.84];
C = [1 -0.6 -0.9];

brus = randn(N+extraN,1);

y1 = filter(C, 1, brus); y1 = y1(extraN: end);
y2 = filter(1, A, brus); y2 = y2(extraN: end);
y3 = filter(C, A, brus); y3 = y3(extraN: end);

%Plot figures
figure
subplot(311)
plot(y1)
title('Time-domain')
ylabel('MA process')
subplot(312)
plot(y2)
ylabel('AR process')
subplot(313)
plot(y3)
ylabel('ARMA process')
xlabel('Time')

%ACF
noLags = 20;
figure
subplot(311);
acf(y1, noLags, 0.05, 1); 
title('ACF');
ylabel('MA process')
subplot(312);
acf(y2, noLags, 0.05, 1); 
ylabel('AR process');
subplot(313);
acf(y3, noLags, 0.05, 1);
ylabel('ARMA process')

%PACF
figure
subplot(311)
pacf( y1, noLags, 0.05, 1 );
title('PACF')
subplot(312)
pacf( y2, noLags, 0.05, 1 );      
subplot(313)
pacf( y3, noLags, 0.05, 1 ); 

%% Q2

rootsA = roots(A);
rootsC = roots(C);



