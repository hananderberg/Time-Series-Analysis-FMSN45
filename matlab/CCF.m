function [] = CCF(ey,ex,M)
    if nargin <3
        M = 50;
    end
    
    %Tänk att y är residual
    
    figure
    [Cxy,lags] = xcorr(ey, ex, M, 'coeff' );
    stem( lags, Cxy )
    hold on
    condInt = 2*ones(1,length(lags))./sqrt( length(ey) );
    plot( lags, condInt,'r--' )
    plot( lags, -condInt,'r--' )
    hold off
    xlabel('Lag')
    ylabel('Amplitude')
    title('Crosscorrelation')
end

