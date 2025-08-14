%%% disp mask
clear
close all
tic

%% load
win_size = 5
thr = 0.03
load lkh_win_5.mat
load gazedata


%% from all session
session_all = [];
N_value = [3, 3, 3, 3, 3];
S = numel(session);
for s = 1 : S
    tmp = semantic_attribute_seq{s};
    tmp(2, :) = tmp(2, :) + 3; % attribute type 2: value range [ 5 -- 8 ]
    tmp(3, :) = tmp(3, :) + 6; % attribute type 3: value range [ 9 -- 12]
    tmp(4, :) = tmp(4, :) + 9; % attribute type 4: value range [ 13 -- 17]
    tmp(5, :) = tmp(5, :) + 12; % attribute type 4: value range [ 13 -- 17]
    tmp2 = zeros(sum(N_value), size(tmp, 2));
    for t = 1 : size(tmp2, 2)
        tmp2(tmp(:, t), t) = 1;
    end
    session_all = [session_all; {tmp2}]; % concatenate segments
end

[ seglabel, seqmask ] = calc_segments(lkh_ses, semantic_attributes, semantic_attribute_seq, win_size, thr);

ses = [17]
for s = ses
    figure('Name', sprintf('participant:%d, task:%d', session{s}.usr, session{s}.task))
    colormap gray
    subplot(2, 1, 1)
    sa = [session_all{s}(1:3, :); 0.5*ones(1, length(session_all{s}(1, :)));session_all{s}(4:6, :); 0.5*ones(1, length(session_all{s}(1, :)));session_all{s}(7:9, :); 0.5*ones(1, length(session_all{s}(1, :)));session_all{s}(10:12, :); 0.5*ones(1, length(session_all{s}(1, :)));session_all{s}(13:15, :)];
    imagesc(sa)
    subplot(2, 1, 2)
    sm = [seqmask{s}(1:3, :); 0.5*ones(1, length(seqmask{s}(1, :)));seqmask{s}(4:6, :); 0.5*ones(1, length(seqmask{s}(1, :)));seqmask{s}(7:9, :); 0.5*ones(1, length(seqmask{s}(1, :)));seqmask{s}(10:12, :); 0.5*ones(1, length(seqmask{s}(1, :)));seqmask{s}(13:15, :)];
    imagesc(sm)
end
toc
seqmask = reshape(seqmask, size(session));
%save('seqmask', 'seqmask')

figure
for i = 1 : 5
  subplot(5, 1, i)
  imagesc(session_all{17}((i-1)*3+1:i*3, :))
  colormap gray
end
figure
for i = 1 : 5
  subplot(5, 1, i)
  imagesc(seqmask{17}((i-1)*3+1:i*3, :))
  colormap gray
end
