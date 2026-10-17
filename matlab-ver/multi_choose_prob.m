function [ out ] = multi_choose_prob(prob, N_hist)
%%% out = exact_prob(prob, N_hist)
%%% prob: multinominal probability distribution: sum(prob) == 1
%%% N_hist: histogram of each observation

if length(prob) ~= length(N_hist)
    error('length of prob must be same as the length of N_hist')
end
if max(N_hist)>1 || min(N_hist) < 0
    error('elements of N_hist must be {0, 1}')
end
prob_vec = prob(:); % used as given (not renormalized)
x = N_hist(:);

N = sum(N_hist);
allpair = combnk(1:length(prob_vec), N);
pairprob = zeros(size(allpair, 1), 1);
for i = 1 : size(allpair, 1)
    pairprob(i) = prod(prob(allpair(i, :)));
    if sum(N_hist(allpair(i, :)) - ones(1, N))==0
        mi = i;
    end
end
pairprob = normalize(pairprob);

out = pairprob(mi);