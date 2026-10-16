function [ out ] = exact_prob(prob, N_hist)
%%% out = exact_prob(prob, N_hist)
%%% prob: multinominal probability distribution: sum(prob) == 1
%%% N_hist: histogram of each observation

if length(prob) ~= length(N_hist)
    disp('length of prob must be same as the length of N_hist')
    return
end
if sum(prob) ~= 1
    prob_vec = normalize(prob);
end    

prob_vec = prob(:);
x = N_hist(:);

N = sum(N_hist);
out = factorial(N) * prod((prob_vec .^ x) ./ factorial(x));
