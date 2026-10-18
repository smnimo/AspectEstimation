%%% learn aspects from the attributes-of-focus (step 2 of the paper)
%%% input: gazedata.mat, lkh_win_5.mat (calc_lkh.m)
%%% output: segsessions.mat, results/p_{spe,all}_topicK.mat, results/Akq_{spe,all}_topicK.csv
clear
close all
tic
%% load
win_size = 5
load lkh_win_5.mat
load gazedata

%% saveparam
savedir = 'results';
if ~exist(savedir, 'dir')
    mkdir(savedir);
end

%% Parameter for Learn aspects
train_num = 100;
lim = 1e-4;
itr_MAX = 1000;
seed = 0; % random seed of the restarts (train.m); [] = not seeded (as for the paper)

%% print param
S = numel(lkh_ses);
thr = 0.03
N_value = max(cell2mat(semantic_attributes(:)))'; 
N_att = size(cell2mat(semantic_attributes(:)), 2);
attrange = cell(1, N_att);
for n = 1 : N_att
    attrange{n} = 1 : N_value(n); % attribute value range
end

%% all attribute values (without the AOF step): one-hot (sum(N_value) x T)
% feature index of attribute a and value v: sum(N_value(1:a-1)) + v
offset = [0; cumsum(N_value(1:end-1))];
session_all = cell(S, 1);
for s = 1 : S
    tmp = semantic_attribute_seq{s} + repmat(offset, 1, size(semantic_attribute_seq{s}, 2));
    tmp2 = zeros(sum(N_value), size(tmp, 2));
    for t = 1 : size(tmp2, 2)
        tmp2(tmp(:, t), t) = 1;
    end
    session_all{s} = tmp2;
end

%% AOF masks (always recomputed; a saved segsessions.mat is not reused)
segsession = cell(S, 1);
[ seglabel, seqmask ] = calc_segments(lkh_ses, semantic_attributes, semantic_attribute_seq, win_size, thr);
for s = 1 : S
    segsession{s} = [];
    if max(seglabel{s}) > 0 % if distinctive periods exist
        masked_session = session_all{s} .* seqmask{s};
        for n = 1 : max(seglabel{s})
            tmp = masked_session(:, seglabel{s}==n);
            tmp(:, sum(tmp)==0) = [];
            if ~isempty(tmp)
                segsession{s} = [segsession{s}, tmp];
            else
                seglabel{s}(seglabel{s}==n) = 0;
            end
        end
        % renumber the remaining segments 1, 2, ... without gaps
        l = seglabel{s};
        [~, ~, ic] = unique(l(l > 0));
        l(l > 0) = ic;
        seglabel{s} = l;
    end
end
save('segsessions.mat', 'segsession', 'session_all', 'seglabel', 'seqmask', 'win_size', 'thr');

%% time points without AOF are not observations
for s = 1 : S
    seqmask{s}(:, sum(seqmask{s})==0) = [];
end
%% learn topic
for topic = 1:5
fprintf('topic:%d\n', topic);
%% from mask
[p, L] = train(seqmask, topic, lim, train_num, itr_MAX, seed); %learn aspects by em
% save
save(sprintf('./%s/p_spe_topic%d.mat', savedir, topic), 'p', 'L');
csvwrite(sprintf('./%s/Akq_spe_topic%d.csv', savedir, topic), cell2mat(p.vz));

%% from all
[p, L] = train(session_all, topic, lim, train_num, itr_MAX, seed); % learn aspects by em
% save
save(sprintf('./%s/p_all_topic%d.mat', savedir, topic), 'p', 'L');
csvwrite(sprintf('./%s/Akq_all_topic%d.csv', savedir, topic), cell2mat(p.vz));

end
toc
