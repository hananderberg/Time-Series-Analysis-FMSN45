% FUNCTION triplePlot plots acf, pacf, and normplot(qq)

function triplePlot(Y,N,includeZero)
if nargin <3
    includeZero = true;
end

if nargin <2
    N = 20;
end

subplot(2,2,1)
acf(Y, N, 0.05,true, [],includeZero);
title('ACF')
subplot(2,2,2)
pacf(Y,N,0.05,true,includeZero);
title('PACF')
subplot(2,2,3:4)
plot(Y);
title('Realization')


end

