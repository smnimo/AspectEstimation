%%% learn aspects --> plot transition of interests
clear
close all
addpath('../functions')
tic
%% load
load gazedata

%% saveparam
savedir = 'results';

%% Parameter for Learn aspects
train_num = 100;
lim = 1e-4;
itr_MAX = 1000;

%% print param
S = numel(session);
thr = 0.03
N_value = max(cell2mat(semantic_attributes(:)))'; 
N_att = size(cell2mat(semantic_attributes(:)), 2);
attrange = cell(1, N_att);
for n = 1 : N_att
    attrange{n} = 1 : N_value(n); % attribute value range
end

session_all = cell(S, 1);
for s = 1 : S
    tmp = semantic_attribute_seq{s};
    tmp(2, :) = tmp(2, :) + 3; % attribute type 2: value range [ 4 -- 6 ]
    tmp(3, :) = tmp(3, :) + 6; % attribute type 3: value range [ 7 -- 9]
    tmp(4, :) = tmp(4, :) + 9; % attribute type 4: value range [ 10 -- 12]
    tmp(5, :) = tmp(5, :) + 12; % attribute type 4: value range [ 13 -- 15]
    tmp2 = zeros(sum(N_value), size(tmp, 2));
    for t = 1 : size(tmp2, 2)
        tmp2(tmp(:, t), t) = 1;
    end
    session_all{s} = tmp2; % concatenate segments
end


%% learn topic
topic = 3

%% from all
N_part = 10;
for part = 1 : N_part
    session_part = cell(S, 1);
    for s = 1 : S
        L = size(session_all{s}, 2);
        session_part{s} = session_all{s}(:,  (part-1) * floor((L/N_part))+1 : part*floor(L/N_part));
    end
    [p, L] = train(session_part, topic, lim, train_num, itr_MAX); % learn aspects by em
    %figure
    %imagesc(p.rz') % plot aspects'
    %colormap gray
    % save
    save(sprintf('./%s/p_part%d-%d_topic%d.mat', savedir, part, N_part, topic), 'p', 'L');
    csvwrite(sprintf('./%s/Akq_part%d-%d_topic%d.csv', savedir, part, N_part, topic), cell2mat(p.vz)); %'
end
toc