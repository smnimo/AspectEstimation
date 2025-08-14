function [ out ] = multinomial(N, varargin)

if N ~= sum(cell2mat(varargin))
    disp('N must be sum of varagin')
    return
end

out = factorial(N) / prod(factorial(cell2mat(varargin)));
