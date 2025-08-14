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


for s = [17,34,46]
    figure('Name', sprintf('participant:%d, task:%d', session{s}.usr, session{s}.task))
    subplot(2, 1, 1)
    imagesc(session_all{s})
    subplot(2, 1, 2)
    imagesc(seqmask{s})
    hold on
    plot(1:length(session_all{s}), ones(1, length(session_all{s})) * 3 + 0.5)
    plot(1:length(session_all{s}), ones(1, length(session_all{s})) * 6 + 0.5)
    plot(1:length(session_all{s}), ones(1, length(session_all{s})) * 9 + 0.5)
    plot(1:length(session_all{s}), ones(1, length(session_all{s})) * 12 + 0.5)
end
