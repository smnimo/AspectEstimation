%%% match the learned aspects with the given tasks (aspect k <-> task k) and reorder p.vz / p.zs
%%% input: task_att.mat (make_task_att.m), results/p_{spe,all}_topicK.mat (learn_aspects_from_mask.m)
%%% output: the same files (reordered, with the permutation 'perm') and results/Akq_{spe,all}_topicK.csv
%%% (the matching of learn_aspects_from_part.m results is in learn_aspects_from_part.m)
clear
load task_att
savedir = 'results'

targets = {'spe', 'all'};
for ti = 1 : numel(targets)
for topic = 2:5
    fname = sprintf('./%s/p_%s_topic%d.mat', savedir, targets{ti}, topic);
    load(fname);

    allpair = sortrows(perms(1:topic)); % lexicographic order: an already matched p keeps its order (idempotent)

    asp = cell2mat(p.vz);
    corscore = zeros(size(allpair, 1), 1);
    for i = 1 : size(allpair, 1)
    cor_task = pdist([normalize(giventask, 2); asp(:, allpair(i, :))'], 'cosine');
    cor_task = 1 - squareform(cor_task);
    cor_task = cor_task(1:size(giventask, 1), end-topic+1:end);%row:given task, col: learned aspects
    corscore(i) = sum(diag(cor_task));
    end
    [~, mi] = max(corscore); % first maximum
    perm = allpair(mi, :);
    p.vz = p.vz(perm);
    p.zs = p.zs(:, perm);
    save(fname, 'p', 'L', 'perm');
    csvwrite(sprintf('./%s/Akq_%s_topic%d.csv', savedir, targets{ti}, topic), cell2mat(p.vz));
end
end
