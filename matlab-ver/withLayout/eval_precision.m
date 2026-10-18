function [ patk_region, patk_item ] = eval_precision(session, pzs, pvz, K)
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
[~, item_number] = sort(pis, 'descend');
topK = item_number(1 : K);

patk_item = zeros(K, 1);
for i = 1 : K
    patk_item(i) = sum(session.answer(topK(1 : i))) / i;
end

prs = pzs * pvz * session.prv;
[~, item_number] = sort(prs, 'descend');
topK = item_number(1 : K);

session_hist = hist(session.f, 1 : 12);
[~, focused_item] = sort(session_hist, 'descend');
answer = zeros(size(session.prv, 2), 1);
answer(focused_item(1 : K)) = 1;
patk_region = zeros(K, 1);
for i = 1 : K
    patk_region(i) = sum(answer(topK(1 : i))) / i;
end

