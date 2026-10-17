function [ p, L ] = train(session, topic, lim, train_num, itr_MAX, seed)
% [ p, L ] = train(session, topic, lim, train_num, itr_MAX, seed)
% best of train_num random restarts of plsa.m (by the final log-likelihood)
% seed (optional): restart i uses rng(seed + i); omitted or [] = not seeded (not reproducible)
if nargin < 6
    seed = [];
end

p = cell(train_num, 1);
tmpL = zeros(train_num, 1);
L = cell(train_num, 1);
parfor i = 1 : train_num
%for i = 1 : train_num
	fprintf('i = %d\n', i);
	if ~isempty(seed)
		rng(seed + i, 'twister');
	end
	[p{i}, lkhlog] = plsa(session, topic, lim, itr_MAX);
    tmpL(i) = lkhlog(end);
    L{i} = lkhlog;
end
[~, mi] = max(tmpL);
p = p{mi};
L = L{mi};

