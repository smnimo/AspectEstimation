clear all
close all

%%% load data
h = './data/mat/'; % place of dataset
session = struct;
load ./area_p.mat;
load ./data/content.mat;
S = 81; % the number of sessions

%%% parameter setting
train_num = 100;
train_lim = 10^(-2);
est_lim = 10^(-3);
naspects = [3 : 3 : 12];

%alpha_num = [0.5, 0];
alpha = 0.5;
attr = 1;
type = 2;
%for alpha = alpha_num
%% 
%%% load data

fid = fopen('./data/experiment/timestamp.list');
for s = 1 : S
	tline = fgetl(fid);
	load(strcat(h, tline, '.mat'));% load f and list, answer, content_id, tadk_id
 	item = itemlist(3, 4, list);
 	[session(s).f, session(s).col] = colseq(f, item.list, content.colhist);
    session(s).piv = item_attr(content, item, attr);
 	session(s).prv = reg_attr(content, item, area.p', alpha, attr, type);
    session(s).list = list;
    session(s).content_id = content_id;
    session(s).task_id = task_id;
    session(s).answer = zeros(length(list.index), 1);
    session(s).answer(answer) = 1;
end
fclose(fid);

%% 
%%% training except one task and evaluation

p = cell(max(naspects), 1);

for topic = naspects
    fprintf('topic:%d\n',topic);
    p{topic} = train(session, topic, train_lim, train_num);
end

        
%% 
%%% save
%save(sprintf('./result/P_alpha%d', alpha * 100), 'p');
save('./result/P_beta', 'p');

%end
