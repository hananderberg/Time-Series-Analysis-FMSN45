function [y] = simulateMyARMA(C, A, N, sigma2, omitted)
    rng (0);
    e = sqrt (sigma2) * randn(N+omitted, 1);
    y1 = filter(C, A, e);
    y = y1(omitted+1: end);
end

