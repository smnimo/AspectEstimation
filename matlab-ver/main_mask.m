clear
close all

%ReadwithTimestamp   % raw gaze data (../experiment, ../knowledge) -> gazedata.mat
calc_lkh            % -> lkh_win_5.mat
%disp_mask
learn_aspects_from_mask % -> segsessions.mat, results/p_{spe,all}_topicK.mat

make_task_att       % -> task_att.mat (paper Table 1)
calc_matching
CalcSim('results');
plot_conf_matrix('results');
