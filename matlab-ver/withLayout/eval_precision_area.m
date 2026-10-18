function [ patk ] = eval_precision_area(session, pzs, pvz, K)
% patk = eval_prevision(session, pzs, pvz)
% patk    : score based on precision
% session : session data with content_id, task_id, answer
% pzs     : p(z|s) of new session, learned by est.m
% pvz     : p(v|z) of all session, learned by train.m
% we estimate interest of every item and evaluate by precision

%colhist = content.colhist;
%list = session.list;

%%% make p(i|s)
%sep_N = size(content.colhist{1}{1}, 1);
%item_colhist = zeros(length(list.index), size(colhist{1}{1}, 2));
% for i = 1 : length(list.index)
%     if size(colhist{list.index(i)}{list.num(i)}, 1) == 1
%         item_colhist(i, :) = colhist{list.index(i)}{list.num(i)};
%     else
%         item_colhist(i, :) = sum(colhist{list.index(i)}{list.num(i)});
%     end
% end
% all_colhist = sum(item_colhist);

% pis = zeros(length(list.index), 1);
% for i = 1 : length(list.index)
%     for n = 1 : sep_N
%         pis(i) = pis(i) + pzs * pvz *  session.prv(:, i); (colhist{list.index(i)}{list.num(i)}(n, :) ./ (all_colhist + (all_colhist == 0)))'];
%     end
% end

pis = pzs * pvz * session.prv;

%% 
% precision
% p@K
patk = zeros(K, 1);
[~, item_number] = sort(pis, 'descend');
topK = item_number(1 : K);

session_hist = hist(session.f, 1 : 12);
[~, focused_item] = sort(session_hist, 'descend');
answer = zeros(size(session.prv, 2), 1);
answer(focused_item(1 : K)) = 1;

for i = 1 : K
    patk(i) = sum(answer(topK(1 : i))) / i;
end
