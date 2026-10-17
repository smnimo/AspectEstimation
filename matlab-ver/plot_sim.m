function plot_sim(loaddir)
%%% plot similarity
%clear
%close all

try
    load(sprintf('./%s/cor.mat', loaddir));
catch
    CalcSim(loaddir)
    load(sprintf('./%s/cor.mat', loaddir));
end

topicrange = 3;
rangelength = (max(topicrange) - min(topicrange))/2 + 1;

% plsa: task-aspect
corta = cellfun(@max, cor, 'UniformOutput', false);
corta_a = cellfun(@max, cor_all, 'UniformOutput', false);
% plsa: aspect-task
cortt = cellfun(@(x) max(x, [], 2)', cor, 'UniformOutput', false);
cortt_a = cellfun(@(x) max(x, [], 2)', cor_all, 'UniformOutput', false);
%{
figure('Name', 'TopicModel: aspect-task')
%{
for topic = topicrange
    for tt = 0 : 2 : 4
subplot(rangelength, 3, (topic/2 - 2) * 3 + tt / 2 + 1)
bar([corta{tt / 2 + 1, topic}; corta_a{tt / 2 + 1, topic}])
set(gca,'xticklabel',{'segmentation','all'})
title(sprintf('topic:%d, tasknum:%d', topic, 8-tt));
ylabel('Cosine similarity')
    end
end
%}
topic = 3
bar([corta{topic}; corta_a{topic}])
set(gca,'xticklabel',{'with MET','without MET'})
title('aspect to task');
ylabel('Cosine similarity')
ylim([0 1])
score{1} = mean([corta{topic}; corta_a{topic}], 2)';
%{
subplot(rangelength, 2, (topic - 2))
bar(mean([corta{tt / 2 + 1, topic}; corta_a{tt / 2 + 1, topic}], 2))
hold on
errorbar(mean([corta{tt / 2 + 1, topic}; corta_a{tt / 2 + 1, topic}], 2), [std(corta{tt / 2 + 1, topic}); std(corta_a{tt / 2 + 1, topic})], '.')
set(gca,'xticklabel',{'with preprocessing','without preprocessing'})
title(sprintf('topic:%d, # of candidate:%d', topic, topic/2-2));
ylabel('Cosine similarity')
%}
%}
figure('Name', 'correlation between given task and learned aspects')
subplot(1, 2, 1)
bar([cortt{topic}; cortt_a{topic}])
set(gca,'xticklabel',{'with MET','without MET'})
title('correlations between given task and learned aspects');
legend('task 1', 'task 2', 'task 3')
ylabel('Cosine similarity')
ylim([0 1])
score{2} = mean([cortt{topic}; cortt_a{topic}], 2)';

subplot(1, 2, 2)
bar(mean([cortt{topic}; cortt_a{topic}], 2))
hold on
errorbar(mean([cortt{topic}; cortt_a{topic}], 2), [std(cortt{topic}); std(cortt_a{topic})], '.') % std (was var)
set(gca,'xticklabel',{'with MET','without MET'})
title('mean of correlations');
ylabel('Cosine similarity')
ylim([0 1])


