function [e_hat] = myFilter(A, C, y)
    e = filter(A, C, y);
    e_hat = e(length(A) :end);
end


