function [ p ] = mstep(gamma, O)
% [ prob(s) ] = mstep(gamma, session)
%
% p(s)      : p.vz{cell} = multinomial, pzs
% session(s): data-set session, prv, prvt
% gamma     : structure: post probability P(v, z|r_t, r_t-1, s)
%

p = struct;
S = length(gamma);
N = size(cell2mat(O'), 1);
K = size(gamma{1}, 1);

rz = zeros(K, N);
for s = 1 : S
    if isempty(gamma{s})
        p.zs(s, :) = normalize(ones(1, K));
    else
        p.zs(s, :) = normalize(sum(gamma{s}, 2)');
        for n = 1 : N
            rz(:, n) = rz(:, n) + sum(gamma{s}(:, O{s}(n, :)==1), 2);
        end
    end
end
sumE = sum(rz, 2);
tmp = rz ./ repmat(sumE, 1, N);
for k = 1 : K
    p.vz{k} = tmp(k, :)';
end
