function [ p, L ] = train(session, topic, lim, train_num, itr_MAX)
% prob(o) = train(session(s), topic)
% prob(o)   :
% session(s):
% topic     :

p = cell(train_num, 1);
tmpL = zeros(train_num, 1);
L = cell(train_num, 1);
parfor i = 1 : train_num
%for i = 1 : train_num
	fprintf('i = %d\n', i);
	[p{i}, lkhlog] = plsa(session, topic, lim, itr_MAX);
    tmpL(i) = lkhlog(end);
    L{i} = lkhlog;
end
[~, mi] = max(tmpL);
p = p{mi};
L = L{mi};

