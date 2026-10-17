function [cor_task, cortop] = CalcEval(session, topic, loaddir, target)
%% load
load(sprintf('./%s/p_%s_topic%d', loaddir, target, topic));

%% calc task
load task_att
cor_task = pdist([normalize(giventask, 2); cell2mat(p.vz)'], 'cosine');
cor_task = 1 - squareform(cor_task);
cor_task = cor_task(1:size(giventask, 1), end-topic+1:end);%row:given task, col: learned aspects

%% sessions without AOF (uniform p.zs in the 'spe' model) are excluded (NaN)
S = numel(session);
noaof = false(S, 1);
if strcmp(target, 'spe')
    tmp = load('segsessions', 'seqmask');
    noaof = cellfun(@(m) ~any(m(:)), tmp.seqmask);
end

cortop = nan(1, S);
for s = 1 : S
if ~noaof(s)
    cor = cor_task(session{s}.task, :);
    [~, mi] = max(p.zs(s, :));
    cortop(s) = cor(mi);
end
end