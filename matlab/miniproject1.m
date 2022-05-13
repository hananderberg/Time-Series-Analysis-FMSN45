%Q1 - Listen to audiofile
[y,Fs]=audioread('fa.wav');
plot(y);
%sound(y, Fs);

%Q2 - Extract 200 samples
y_extracted = y(5500:5699);
plot(y_extracted);
rho = acf(y_extracted, 100);
plot(rho);

%Q3 - Plot the periodogram
X=fft(y_extracted); 
n=length(y_extracted);
Rhat=(X.*conj(X))/n;
f=[0:n-1]/n;
plot(f,Rhat);


