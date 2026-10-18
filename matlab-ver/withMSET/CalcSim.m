function [cor, cortop, cor_all, cortop_all] = CalcSim(loaddir)

%% load
load lkh_win_5.mat
load gazedata


%%
cor = cell(5, 1);
cortop = cell(5, 1);
cor_all = cell(5, 1);
cortop_all = cell(5, 1);

for topic = 1:5
fprintf('topic : %d\n', topic);

[cor{topic}, cortop{topic}] = CalcEval(session, topic, loaddir, 'spe');
[cor_all{topic}, cortop_all{topic}] = CalcEval(session, topic, loaddir, 'all');
end

%%
save(sprintf('./%s/cor.mat', loaddir), 'cor', 'cortop', 'cor_all', 'cortop_all');