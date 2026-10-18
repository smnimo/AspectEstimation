clear all
close all

%%% leave-one-out evaluation (paper 4.2-4.4, Fig. 5-7): learn the aspects from all sessions but one, estimate the
%%% interests from the first part of that session and predict the gazed regions / selected items of the rest
%%% input: data/mat/*.mat, data/content.mat, data/experiment/timestamp.list, area_s.mat (make_area.m)
%%% output: result/{region,item}/MP_attr<attr>_alpha<100 alpha>.mat (mpatk{aspects, ratio}: P@1..3)

%%% load data
h = './data/mat/'; % place of dataset
session = struct;
load ./area_s.mat; % layout effect: gaze frequency of all sessions but one (make_area.m)
load ./data/content.mat;
S = 81; % the number of sessions

%%% parameter setting
train_num = 100;
train_lim = 10^(-2);
est_lim = 10^(-3);
naspects = [3 : 3 : 12];

alpha_num = [0, 0.5]; % mixture ratio of the layout effect (paper: 0, 0.5; Fig. 6: 0 : 0.1 : 1)
attr_num = [1, 2, 3]; % 1: semantic (SA), 2: color (AA), 3: both (SA+AA)

for attr = attr_num
for alpha = alpha_num
%% 
%%% load data

fid = fopen('./data/experiment/timestamp.list');
for s = 1 : S
	tline = fgetl(fid);
	load(strcat(h, tline, '.mat'));% load f and list, answer, content_id, tadk_id
 	item = itemlist(3, 4, list);
 	[session(s).f, session(s).col] = colseq(f, item.list, content.colhist);
    session(s).piv = item_attr(content, item, attr);
    session(s).list = list;
    session(s).content_id = content_id;
    session(s).task_id = task_id;
    session(s).answer = zeros(length(list.index), 1);
    session(s).answer(answer) = 1;
    session(s).content = content;
    session(s).item = item;
end
fclose(fid);

%% 
%%% training except one task and evaluation

%p = cell(max(naspects), all_task_num);
patk_r = cell(max(naspects), 9);
patk_i = cell(max(naspects), 9);
mpatk_r = cell(max(naspects), 9);
mpatk_i = cell(max(naspects), 9);
K = 3; % P@k
for topic = naspects
    for r = 1 : 9
        patk_r{topic, r} = zeros(K, S);
        patk_i{topic, r} = zeros(K, S);
    end
end

for topic = naspects
    for s = 1 : S
     	fprintf('topic:%d\n',topic);
        fprintf('session:%d\n', s);
        train_session = session;
        train_session(s) = [];
        test_session = session(s);
        for s2 = 1 : S - 1
         	train_session(s2).prv = reg_attr(train_session(s2).content, train_session(s2).item, area_s(:, s)', alpha, attr);
        end
        test_session.prv = reg_attr(test_session.content, test_session.item, area_s(:, s)', alpha, attr);
        p = train(train_session, topic, train_lim, train_num);
        T = length(test_session.f);
        session_est = test_session;
        session_test = test_session;
        for ratio = 1 : 9
            session_est.f = test_session.f(1 : floor(ratio * T / 10));
            session_test.f = test_session.f(ceil(ratio * T / 10) : T);
            pzs = est(p, session_est, est_lim);
            [patk_r{topic, ratio}(:, s), patk_i{topic, ratio}(:, s)] = eval_precision(session_test, pzs, p.vz, K);
        end
    end
end

for topic = naspects
    for ratio = 1 : 9
        mpatk_r{topic, ratio} = mean(patk_r{topic, ratio}, 2);
        mpatk_i{topic, ratio} = mean(patk_i{topic, ratio}, 2);
    end
end
        
%% 
%%% save
mpatk = mpatk_r;
save(sprintf('./result/region/MP_attr%d_alpha%d.mat', attr, round(alpha * 100)), 'mpatk');
mpatk = mpatk_i;
save(sprintf('./result/item/MP_attr%d_alpha%d.mat', attr, round(alpha * 100)), 'mpatk');

end
end