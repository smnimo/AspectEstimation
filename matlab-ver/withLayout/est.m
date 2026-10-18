function pzsn = est(p, new_session, lim)
% pzsn = est(session(s), prob(o), new(s))
%
% pzsh : new p(z|s) with fold in heuristics
% session(s): data-set session-region, and P(R|V), P(rt|rt-1, V)
% prob(o)   : obj: pvz, pzs, L(log likelyhood)
% new(s)    : new session

K = size(p.zs, 2);

pn = prob();
pn.zs = ones(1, K)/K;
pn.vz = normalize(p.vz + 0.01, 2);

itr = 1;
pn.calcL(new_session);
Lt = pn.L;
cL = 1;
err = 0;

while cL > lim || cL < 0
  %fprintf('itr = %d\n', itr);
  %E-step
  gamma = estep(new_session, pn);
  
  %M-step
  T = size(gamma.pvzr, 3);
  pzs = sum(sum(gamma.pvzr, 3), 2);
  pn.zs = (pzs / T)';
  
  pn.calcL(new_session);
  Lt = [Lt pn.L];
  cL = Lt(itr + 1) - Lt(itr);
  if cL < 0 && itr > 1
    err = err + 1;
  end
  itr = itr + 1;
end
if err > 0
  fprintf('err = %d\n', err);
end

pzsn = pn.zs;
