clear
close all

%ReadwithTimestamp
calc_lkh
%disp_mask
learn_aspects_from_mask

calc_matching
CalcSim('results');
plot_conf_matrix('results');
