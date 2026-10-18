function [ patk ] = eval_precision_item(session, pzs, pvz, K)
% patk = eval_prevision(session, pzs, pvz)
% patk    : score based on precision
% session : session data with content_id, task_id, answer
% pzs     : p(z|s) of new session, learned by est.m
% pvz     : p(v|z) of all session, learned by train.m
% we estimate interest of every item and evaluate by precision

pis = pzs * pvz * session.piv;

%% 
% precision
% p@K
patk = zeros(K, 1);
[~, item_number] = sort(pis, 'descend');
topK = item_number(1 : K);

for i = 1 : K
    patk(i) = sum(session.answer(topK(1 : i))) / i;
end
