function [] = plotBasics(y, noLags)

    if nargin < 2
        noLags = 20;
    end

    figure
    subplot(311)
    acf(y, noLags, 0.05, 1);
    title('ACF')
    subplot(312)
    pacf(y, noLags, 0.05, 1);
    title('PACF')
    subplot(313)
    normplot(y)
    title('Normplot')
    
end

