function [ p ] = mstep(gamma)
% [ prob(o) ] = mstep(session(s), alpha(s))
%
% p(o)      : obj: pvz, pzs, L(log likelyhood)
% session(s): data-set session, prv, prvt
% alpha     : structure: post probability P(v, z|r_t, r_t-1, s)
%

p = prob();
S = length(gamma);
Q = size(gamma(1).pvzr, 2);
pvz = sum(cat(3, gamma.pvzr), 3);
pz = sum(pvz, 2);
p.vz = pvz ./ repmat(pz, 1, Q);%P(v|z)

for s = 1 : S
	T = size(gamma(s).pvzr, 3);
	pzs = sum(sum(gamma(s).pvzr, 3), 2);
	p.zs(s, :) = (pzs / T)';
end
