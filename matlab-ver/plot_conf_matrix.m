function [out_MSET, out_wo_MSET] = plot_conf_matrix(loaddir)
%%% confusion matrix
%clear
%close all

load gazedata
topic = 3;
load(sprintf('./%s/p_spe_topic%d', loaddir, topic));

n_task = size(session, 2);
conf = zeros(n_task, n_task+1);

n_usr = size(session, 1);
for task = 1 : n_task
    [val, est_task] = max(p.zs((task-1) * n_usr + 1: task*n_usr, :),  [], 2);
    est_task(val==1/n_task) = n_task + 1;
    conf(task, :) = hist(est_task, 1:n_task+1);
end
figure('Name', 'Task estimation MSET: confusion matrix');
imagesc(conf)
colormap hot
out_MSET = conf;

load(sprintf('./%s/p_all_topic%d', loaddir, topic));
conf = zeros(n_task, n_task+1);

for task = 1 : n_task
    [val, est_task] = max(p.zs((task-1) * n_usr + 1: task*n_usr, :),  [], 2);
    est_task(val==1/n_task) = n_task + 1;
    conf(task, :) = hist(est_task, 1:n_task+1);
end
figure('Name', 'Task estimation w/o MSET: confusion matrix');
imagesc(conf)
colormap hot
out_wo_MSET = conf;