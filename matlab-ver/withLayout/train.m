function [ p ] = train(session, topic, lim, train_num)
% prob(o) = train(session(s), topic)
% prob(o)   :
% session(s):
% topic     :

p = cell(train_num, 1);
L = cell(train_num, 1);
%for i = 1 : 3
%parfor i = 1 : 100
parfor i = 1 : train_num
%	fprintf('i = %d\n', i);
	p{i} = plsa(session, topic, lim);
    L{i} = p{i}.L;
end
[~, mi] = max(cell2mat(L));
p = p{mi};

