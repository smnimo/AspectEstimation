function [cor_task, cortop] = CalcEval(session, topic, loaddir, target)
%% load
load(sprintf('./%s/p_%s_topic%d', loaddir, target, topic));

%% calc task
load task_att
cor_task = pdist([normalize(giventask, 2); cell2mat(p.vz)'], 'cosine');
cor_task = 1 - squareform(cor_task);
cor_task = cor_task(1:size(giventask, 1), end-topic+1:end);%row:given task, col: learned aspects

cortop = [];
corall = [];

%%
S = numel(session);
for s = 1 : S

%% plot
cor = cor_task(session{s}.task, :);
[~, mi] = max(p.zs(s, :));
cortop = [cortop, cor(mi)];
%corall = [corall, cor * normalize(area_int)];
end