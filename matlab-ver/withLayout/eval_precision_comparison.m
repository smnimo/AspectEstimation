function [patk_region, patk_item] = eval_precision_comparison(session, content, attr, type, K)
%????????
attr_set = makeattr(1 : 12, content, session.list, type);
%cdist = pdist2(attr, attr_set, 'cosine');
%bhattacharyya
cdist = zeros(12, 1);
for i = 1 : 12
    cdist(i) = bhattacharyya(attr, attr_set(i, :));
end
[~, item_number] = sort(cdist);
topK = item_number(1 : K);

patk_item = zeros(K, 1);
for i = 1 : K
    patk_item(i) = sum(session.answer(topK(1 : i))) / i;
end

session_hist = hist(session.f, 1 : 12);
[~, focused_item] = sort(session_hist, 'descend');
answer = zeros(12, 1);
answer(focused_item(1 : K)) = 1;
patk_region = zeros(K, 1);
for i = 1 : K
    patk_region(i) = sum(answer(topK(1 : i))) / i;
end

