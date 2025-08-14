%%% learn aspects --> plot transition of interests
clear
close all
tic
%% load
win_size = 5
load lkh_win_5.mat
load gazedata

%% saveparam
savedir = 'results';

%% Parameter for Learn aspects
train_num = 100;
lim = 1e-4;
itr_MAX = 1000;

%% print param
S = numel(lkh_ses);
thr = 0.03
N_value = max(cell2mat(semantic_attributes(:)))'; 
N_att = size(cell2mat(semantic_attributes(:)), 2);
attrange = cell(1, N_att);
for n = 1 : N_att
    attrange{n} = 1 : N_value(n); % attribute value range
end

%%
try
    load('segsessions')
catch
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

%% make segsession from mask
segsession = cell(S, 1);
[ seglabel, seqmask ] = calc_segments(lkh_ses, semantic_attributes, semantic_attribute_seq, win_size, thr);
mods = [];
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
                mods = [mods, s];
            end
        end
    end
end

for s = 1 : length(mods)
    l = seglabel{mods(s)};
    for n = 1 : max(l)
        if (nnz(l==n)==0) && (n < max(l))
            l(l>n) = l(l>n)-1;
            n = n - 1;
        end
    end
    seglabel{mods(s)} = l;
end
save('segsessions', 'segsession', 'session_all', 'seglabel', 'seqmask', 'win_size', 'thr');
end
for s = 1 : S
    seqmask{s}(:, sum(seqmask{s})==0) = [];
end
%% learn topic
for topic = 1:5
fprintf('topic:%d\n', topic);
%% from mask
[p, L] = train(seqmask, topic, lim, train_num, itr_MAX); %learn aspects by em
%figure
%imagesc(cell2mat(p.vz)') % plot aspects '
%colormap gray
% save
save(sprintf('./%s/p_spe_topic%d.mat', savedir, topic), 'p', 'L');
csvwrite(sprintf('./%s/Akq_spe_topic%d.csv', savedir, topic), cell2mat(p.vz));

%% from all
[p, L] = train(session_all, topic, lim, train_num, itr_MAX); % learn aspects by em
%figure
%imagesc(p.rz') % plot aspects'
%colormap gray
% save
save(sprintf('./%s/p_all_topic%d.mat', savedir, topic), 'p', 'L');
csvwrite(sprintf('./%s/Akq_all_topic%d.csv', savedir, topic), cell2mat(p.vz)); %'

end
toc