%plot_eval_detectedAoF
clear
close all

load segsessions
load gazedata
load task_att

S = numel(session)
cos_dist_c = zeros(S, 1);
cos_dist_mset = zeros(S, 1);
cos_dist_all = zeros(S, 1);
cos_dist_top = zeros(S, 1);
for s = 1 : S
ci = hist(session{s}.f, 1:12);
[~, mi] = sort(ci, 'descend');
AoF1 = semantic_attributes{s}(mi(1), :);
AoF2 = semantic_attributes{s}(mi(2), :);
AoF = zeros(3, 5);
AoFc = AoF1==AoF2;
for i = 1 : 5
    AoF(AoF1(i), i) = AoFc(i);
end
topval = zeros(3, 5);
for i = 1 : 5
    topval(AoF1(i), i) = 1;
end

cos_dist_c(s) = giventask(session{s}.task, :) / norm(giventask(session{s}.task, :)) * AoF(:) / (norm(AoF(:)) + (norm(AoF(:))==0));
cos_dist_top(s) = giventask(session{s}.task, :) / norm(giventask(session{s}.task, :)) * topval(:) / (norm(topval(:)) + (norm(topval(:))==0));
cos_dist_all(s) = giventask(session{s}.task, :) / norm(giventask(session{s}.task, :)) * sum(session_all{s}, 2) / (norm(sum(session_all{s}, 2)) + (norm(sum(session_all{s}, 2))==0));
cos_dist_mset(s) = giventask(session{s}.task, :) / norm(giventask(session{s}.task, :)) * sum(seqmask{s}, 2) /norm(sum(seqmask{s}, 2));
end

save('distances', 'cos_dist_c', 'cos_dist_mset', 'cos_dist_top', 'cos_dist_all')
cos_dist_c = cos_dist_c(cos_dist_mset>=0);
cos_dist_top = cos_dist_top(cos_dist_mset>=0);
cos_dist_all = cos_dist_all(cos_dist_mset>=0);
cos_dist_mset = cos_dist_mset(cos_dist_mset>=0);

boxplot([cos_dist_all, cos_dist_top, cos_dist_c, cos_dist_mset])
set(gca,'xticklabel',{'All', 'Top', 'Comparison','LD'})
ylabel('Cosine simlarity')
hold on
plot([mean(cos_dist_all), mean(cos_dist_top), mean(cos_dist_c), mean(cos_dist_mset)], 'd')

%%
datas = cell(4, 1);
datas{1} = cos_dist_c;
datas{2} = cos_dist_top;
datas{3} = cos_dist_all;
datas{4} = cos_dist_mset;
res = zeros(4, 4);
resp = zeros(4, 4);
respp = zeros(4, 4);
pairs = combnk(1:4, 2);
for i = 1 : length(pairs)
    ii = pairs(i, 1);
    ij = pairs(i, 2);
    [res(ii, ij), resp(ii, ij)] = ttest2(datas{ii}, datas{ij});
    respp(ii, ij) = resp(ii, ij) < 0.01;
end
