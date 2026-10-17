function [out_MSET, out_wo_MSET] = plot_conf_matrix(loaddir)
%%% confusion matrix
%clear
%close all

load gazedata
tmp = load('segsessions', 'seqmask');
noaof = cellfun(@(m) ~any(m(:)), tmp.seqmask); % sessions without AOF: "no comparison"
topic = 3;
load(sprintf('./%s/p_spe_topic%d', loaddir, topic));

n_task = size(session, 2);
conf = zeros(n_task, n_task+1);

n_usr = size(session, 1);
for task = 1 : n_task
    rows = (task-1) * n_usr + 1: task*n_usr;
    [~, est_task] = max(p.zs(rows, :),  [], 2);
    est_task(noaof(rows)) = n_task + 1; % (their p.zs is uniform; was detected by val == 1/n_task)
    conf(task, :) = hist(est_task, 1:n_task+1);
end
figure('Name', 'Task estimation MSET: confusion matrix');
imagesc(conf)
colormap hot
out_MSET = conf;

load(sprintf('./%s/p_all_topic%d', loaddir, topic));
conf = zeros(n_task, n_task+1);

for task = 1 : n_task
    [~, est_task] = max(p.zs((task-1) * n_usr + 1: task*n_usr, :),  [], 2); % every session has observations
    conf(task, :) = hist(est_task, 1:n_task+1);
end
figure('Name', 'Task estimation w/o MSET: confusion matrix');
imagesc(conf)
colormap hot
out_wo_MSET = conf;