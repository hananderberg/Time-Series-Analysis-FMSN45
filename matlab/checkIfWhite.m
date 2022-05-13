% FUNCTION [deemedWhite] = checkIfWhite( data, [K], [alpha] )
%
% The function computes the Monti test (using K terms, with default K = 2,
% and confidence alpha, with default = 0.05) to determine if the data is
% white nor not, returning the decision. 
%

% Reference: 
%   "An Introduction to Time Series Modeling" by Andreas Jakobsson
%   Studentlitteratur, 2019
%
function deemedWhite = checkIfWhite( data, K, alpha )

if nargin<2
    K=20;
end
if nargin<3
    alpha = 0.05;
end

[deemedWhite, Q, chiV] = montiTest( data, K, alpha );
if deemedWhite
     fprintf('The residual is deemed to be white according to the Monti-test (as %5.2f < %5.2f).\n', Q, chiV );
else
    fprintf('The residual is not deemed to be white according to the Monti-test (as %5.2f > %5.2f).\n', Q, chiV );
end
fprintf('The variance of the residual is %2.6f.\n', var(data) )
