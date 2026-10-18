function [ alpha ] = estep(session, p)
% [ alpha(s) ] = estep(session(s), p(o))
%
% alpha     : structure: post probability P(v, z|r_t, r_t-1, s)
% session(s): data-set session, prv, prvt
% p(o)      : obj: pvz, pzs, L(log likelyhood)
%

vz = p.vz;
zs = p.zs;

S = length(session);
[K, Q] = size(p.vz);

alpha = struct;
for s = 1 : S
	prv = session(s).prv;
	f = session(s).f;
	T = length(f);
	pvzr = zeros(K, Q, T); %P(v, z|rt, s);
	%pvzrt = (zs(s, :)' * prv(:, f(t))') .* vz;
    for t = 1 : T
%    	pvzrt = (zs(s, :)' * [prv(:, f(t)); session(s).col(t, :)']') .* vz;
    	pvzrt = (zs(s, :)' * prv(:, f(t))') .* vz;
    	pvzr(:, :, t) = pvzrt / sum(pvzrt(:));
	end
	alpha(s).pvzr = pvzr;
end
