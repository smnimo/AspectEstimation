%%% learn the aspects and the interests from all sessions (paper 4.4.3, Fig. 8, 9)
%%% input: data/mat/*.mat, data/content.mat, data/experiment/timestamp.list
%%% output: result/P_attr<attr>_alpha<100 alpha>.mat (vz{aspects} = P(V|Z), zs{aspects} = P(Z|s), L{aspects}, layout)
clear all
close all

%%% load data
h = './data/mat/'; % place of dataset
session = struct;
load ./data/content.mat;
S = 81; % the number of sessions

%%% parameter setting
train_num = 100;
train_lim = 10^(-2);
naspects = [3 : 3 : 12];
alpha = 0.5; % mixture ratio of the layout effect
attr = 1;    % 1: semantic (SA), 2: color (AA), 3: both (SA+AA)

%%% load data
fid = fopen('./data/experiment/timestamp.list');
items = cell(S, 1);
counts = zeros(1, 12);
for s = 1 : S
	tline = fgetl(fid);
	load(strcat(h, tline, '.mat'));% load f and list, answer, content_id, tadk_id
 	items{s} = itemlist(3, 4, list);
 	[session(s).f, session(s).col] = colseq(f, items{s}.list, content.colhist);
    session(s).piv = item_attr(content, items{s}, attr);
    session(s).list = list;
    session(s).content_id = content_id;
    session(s).task_id = task_id;
    session(s).answer = zeros(length(list.index), 1);
    session(s).answer(answer) = 1;
    counts = counts + hist(session(s).f', 1 : 12);
end
fclose(fid);
layout = normalize(counts); % layout effect: relative gaze frequency of all sessions (paper Eq. 11)
for s = 1 : S
 	session(s).prv = reg_attr(content, items{s}, layout, alpha, attr);
end

%%% learn the aspects
vz = cell(max(naspects), 1);
zs = cell(max(naspects), 1);
L = cell(max(naspects), 1);
for topic = naspects
    fprintf('topic:%d\n',topic);
    p = train(session, topic, train_lim, train_num);
    vz{topic} = p.vz; % plain matrices (prob objects cannot be read outside MATLAB)
    zs{topic} = p.zs;
    L{topic} = p.L;
end

%%% save
if ~exist('./result', 'dir')
    mkdir('./result');
end
save(sprintf('./result/P_attr%d_alpha%d.mat', attr, round(alpha * 100)), 'vz', 'zs', 'L', 'layout');
