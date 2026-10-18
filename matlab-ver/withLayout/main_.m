clear all
close all

%%% load data
h = './data/mat/'; % place of dataset
session = struct;
load ./sig.mat;
%load ./area_s.mat;
load ./data/content.mat;
S = 81; % the number of sessions

%%% parameter setting
%train_num = 1;
train_num = 100;
train_lim = 10^(-2);
est_lim = 10^(-3);
naspects = [3 : 3 : 12];
%naspects = [6, 9];
%naspects = 3;

alpha_num = [0.3, 0.7];
attr_num = [1, 2, 3];
%attr_num = 1;
%type_num = [1, 2]; % 1 : sum, 2 : prod
type_num = 1;

for attr = attr_num
for type = type_num
%attr = 1;
%type = 1;
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
         	train_session(s2).prv = reg_attr(train_session(s2).content, train_session(s2).item, area_s(:, s)', alpha, attr, type);
        end
        test_session.prv = reg_attr(test_session.content, test_session.item, area_s(:, s)', alpha, attr, type);
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
save(sprintf('./result/region/MP_attr%d_type%d_alpha%d_beta', attr, type, alpha * 100), 'mpatk');
mpatk = mpatk_i;
save(sprintf('./result/item/MP_attr%d_type%d_alpha%d_beta', attr, type, alpha * 100), 'mpatk');

end
end
end