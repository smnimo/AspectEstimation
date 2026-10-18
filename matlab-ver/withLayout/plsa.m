function [ p ] = plsa(session, K, lim)
% [ prob(o) ] = plsa(session(s), topic)
%
% prob(o)   : obj: pvz, pzs, L(log likelyhood)
% session(s): data-set session-region, and P(R|V), P(rt|rt-1, V)
% topic     : number of principle topic
%

S = length(session);
[Q, N] = size(session(1).prv);
%Q = Q + size(session(1).col, 2);
p = prob();
p.zs = normalize(rand(S, K), 2);
p.vz = normalize(rand(K, Q), 2);

itr = 1;
p.calcL(session);
Lt = p.L;
cL = 1;
err = 0;


while cL > lim || cL < 0
  %fprintf('itr = %d\n', itr);
  %E-step
  gamma = estep(session, p);
  
  %M-step
  p = mstep(gamma);
  
  p.calcL(session);
%  p.L
  Lt = [Lt p.L];
  cL = Lt(itr + 1) - Lt(itr);
  if cL < 0 && itr > 1
    err = err + 1;
  end
  itr = itr + 1;
end
if err > 0
  fprintf('err = %d\n', err);
end
