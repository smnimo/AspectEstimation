clear
close all

%ReadwithTimestamp   % raw gaze data (../experiment, ../knowledge) -> gazedata.mat
calc_lkh            % -> lkh_win_5.mat
%plot_mask(17)      % paper Fig. 7
learn_aspects_from_mask % -> segsessions.mat, results/p_{spe,all}_topicK.mat

make_task_att       % -> task_att.mat (paper Table 1)
calc_matching
CalcSim('results');
plot_conf_matrix('results');
