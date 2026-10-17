function [ p, lkhlog ] = plsa(session, K, train_lim, itr_MAX)
% [ prob ] = plsa(session(s), topic)
%
% prob(o)   : obj: pvz, pzs, L(log likelyhood)
% session(s): data-set session-region, and P(R|V), P(rt|rt-1, V)
% topic     : number of principle topic
%

S = length(session);
p = struct;
N = size(cell2mat(session'), 1);
p.zs = normalize(rand(S, K), 2);
p.vz = cell(K, 1);
for k = 1 : K
    p.vz{k} = normalize(rand(N, 1));
end


lkhlog = [];
inc = -1e20;
lkh_p = -1e20;
itr = 1;

% stops when the relative increase of the log-likelihood is <= train_lim (1e-4 in learn_aspects_from_mask.m);
% the result is not necessarily converged, and lkhlog(end) belongs to the parameters before the last M-step
while abs(inc/lkh_p) > train_lim && itr < itr_MAX
  %fprintf('itr = %d\n', itr);
  %E-step
  [gamma, lkh_r] = estep(session, p);
  
  %M-step
  p = mstep(gamma, session);
    
  inc = lkh_r - lkh_p;
  if inc < 0
    fprintf('error\n');
    break;
  end
  lkhlog = [lkhlog, lkh_r];
  lkh_p = lkh_r;
  itr = itr + 1;
end
