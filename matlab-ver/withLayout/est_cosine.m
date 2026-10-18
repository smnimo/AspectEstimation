function attr = est_cosine(train_session, test_session, K)

test = test_session.allattr;
train = cat(1, train_session.allattr);

%cdist = pdist2(test, train, 'cosine');
%bhattacharyya
cdist = zeros(size(train, 1), 1);
for f = 1 : size(train, 1)
    cdist(f) = bhattacharyya(test, train(f, :));
end
%[~, mi] = min(cdist);
[d, i] = sort(cdist);
attr = zeros(K, length(test));
for k = 1 : K
    attr(k, :) = train(i(k), :) / d(k);
end
attr = normalize(sum(attr, 1));
