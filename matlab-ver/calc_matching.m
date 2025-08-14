clear
load task_att
savedir = 'results'

%%% matching with task and aspects
%{
for topic = 2:5
    load(sprintf('./%s/p_spe_topic%d', savedir, topic));

    allpair = perms(1:topic);

    asp = cell2mat(p.vz);
    corscore = zeros(size(allpair, 1), 1);
    for i = 1 : size(allpair, 1)
    cor_task = pdist([normalize(giventask, 2); asp(:, allpair(i, :))'], 'cosine');
    cor_task = 1 - squareform(cor_task);
    cor_task = cor_task(1:size(giventask, 1), end-topic+1:end);%row:given task, col: learned aspects
    corscore(i) = sum(diag(cor_task));
    end
    [~, mi] = max(corscore);
    p.vz = p.vz(allpair(mi, :));
    p.zs = p.zs(:, allpair(mi, :));
    save(sprintf('./%s/p_spe_topic%d', savedir, topic), 'p', 'L');
    csvwrite(sprintf('./%s/Akq_spe_topic%d.csv', savedir, topic), cell2mat(p.vz));
end

for topic = 2:5
    load(sprintf('./%s/p_all_topic%d', savedir, topic));

    allpair = perms(1:topic);

    asp = cell2mat(p.vz);
    corscore = zeros(size(allpair, 1), 1);
    for i = 1 : size(allpair, 1)
    cor_task = pdist([normalize(giventask, 2); asp(:, allpair(i, :))'], 'cosine');
    cor_task = 1 - squareform(cor_task);
    cor_task = cor_task(1:size(giventask, 1), end-topic+1:end);%row:given task, col: learned aspects
    corscore(i) = sum(diag(cor_task));
    end
    [~, mi] = max(corscore);
    p.vz = p.vz(allpair(mi, :));
    p.zs = p.zs(:, allpair(mi, :));
    save(sprintf('./%s/p_all_topic%d', savedir, topic), 'p', 'L');
    csvwrite(sprintf('./%s/Akq_all_topic%d.csv', savedir, topic), cell2mat(p.vz));
end
%}

N_part = 10
topic = 3
cor_part = cell(3, 1);
for part = 1:N_part
    load(sprintf('./%s/p_part%d-%d_topic%d', savedir, part, N_part, topic));

    allpair = perms(1:topic);

    asp = cell2mat(p.vz);
    corscore = zeros(size(allpair, 1), 1);
    for i = 1 : size(allpair, 1)
    cor_task = pdist([normalize(giventask, 2); asp(:, allpair(i, :))'], 'cosine');
    cor_task = 1 - squareform(cor_task);
    cor_task = cor_task(1:size(giventask, 1), end-topic+1:end);%row:given task, col: learned aspects
    corscore(i) = sum(diag(cor_task));
    end
    [~, mi] = max(corscore);
    p.vz = p.vz(allpair(mi, :));
    p.zs = p.zs(:, allpair(mi, :));
    save(sprintf('./%s/p_part%d-%d_topic%d', savedir, part, N_part, topic), 'p', 'L');
    csvwrite(sprintf('./%s/Akq_part%d-%d_topic%d.csv', savedir, part, N_part, topic), cell2mat(p.vz));
    cor_task_part = pdist([normalize(giventask); cell2mat(p.vz)'], 'cosine');
    cor_task_part = 1 - squareform(cor_task_part);
    cor_part{part} = diag(cor_task_part(1:size(giventask, 1), end-topic+1:end))';%row:given task, col: learned aspects
end
save(sprintf('./%s/cor-%d_topic%d', savedir, N_part, topic), 'cor_part');