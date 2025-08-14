clear
close all


load('./results/cor-5_topic3.mat')
load('./results/cor.mat')

cor = diag(cor{3})'
cor_part = cell2mat(cor_part)

[h, p] = ttest2(cor_part, cor)

subplot(1, 2, 1)
bar(mean(cor_part))
hold on
er = errorbar([1, 2, 3], mean(cor_part),mean(cor_part)-min(cor_part), max(cor_part)-mean(cor_part))
er.LineStyle = 'None'
ylim([0 0.9])
subplot(1, 2, 2)
bar(cor)
ylim([0 0.9])
