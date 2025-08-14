%%%plot_L_with_topics

clear
close all

L_all = zeros(5, 1);
asp_all = cell(5, 1);
L_spe = zeros(5, 1);
asp_spe = cell(5, 1);

for topic = 1 : 5
    load(sprintf('./results/p_all_topic%d', topic));
    L_all(topic) = L(end);
    asp_all{topic} = cell2mat(p.vz);
    
    load(sprintf('./results/p_spe_topic%d', topic));
    L_spe(topic) = L(end);
    asp_spe{topic} = cell2mat(p.vz);
end
figure
subplot(1, 2, 1)
plot(L_all)
title('all')
subplot(1, 2, 2)
plot(L_spe)
title('MSET')

load task_att
figure('Name', 'all')
figure('Name', 'MSET')
for topic = 1 : 5
    figure(2)
    subplot(1, 6, topic)
    imagesc(asp_all{topic})
    colormap gray
    figure(3)
    subplot(1, 6, topic)
    imagesc(asp_spe{topic})
    colormap gray
end
    figure(2)
    subplot(1, 6, 6)
    imagesc(giventask')
    colormap gray
    figure(3)
    subplot(1, 6, 6)
    imagesc(giventask')
    colormap gray
