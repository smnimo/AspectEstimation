function [ gamma, lkh_log ] = estep(session, p)
% [ gamma ] = estep(session(s), p(s))
%

pvz = p.vz;
pzs = p.zs;

S = length(session);
K = size(pzs, 2);

gamma = cell(S, 1); %P(z|vt, s);
r = cell(S, 1);
pvz_s = zeros(K, 1);
for s = 1 : S
    f = session{s};
    T = size(f, 2);
    gamma{s} = zeros(K, T); %P(z|rt, s);
    r{s} = zeros(T, 1);
    for t = 1 : T
        for k = 1 : K
            pvz_s(k) = pzs(s, k) * exact_prob(pvz{k}, f(:, t));
        end
        r{s}(t) = sum(pvz_s);
        gamma{s}(:, t) = pvz_s / r{s}(t);
    end
end
lkh_log = sum(log(cell2mat(r)));
