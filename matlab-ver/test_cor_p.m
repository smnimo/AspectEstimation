clear
close all


%%% compare the aspects learned on temporal parts of the sessions (learn_aspects_from_part.m) with the whole sessions
N_part = 10; % must equal N_part of learn_aspects_from_part.m (refs holds results of runs with 3, 5 and 10 parts)
topic = 3;
load(sprintf('./results/cor-%d_topic%d.mat', N_part, topic))
load('./results/cor.mat')

cor = diag(cor{topic})'
cor_part = cell2mat(cor_part)

[h, p] = ttest2(cor_part, cor)

subplot(1, 2, 1)
bar(mean(cor_part))
hold on
er = errorbar([1, 2, 3], mean(cor_part),mean(cor_part)-min(cor_part), max(cor_part)-mean(cor_part))
set(er, 'LineStyle', 'none') % (works in MATLAB and Octave)
ylim([0 0.9])
subplot(1, 2, 2)
bar(cor)
ylim([0 0.9])
