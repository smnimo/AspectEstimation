clear all
close all

%%% load data
h = './data/mat/'; % place of dataset
session = struct;
%load ./sig.mat;
load ./area_s.mat;
load ./data/content.mat;
S = 81; % the number of sessions

%%% parameter setting
%train_num = 1;
train_num = 100;
train_lim = 10^(-2);
est_lim = 10^(-3);
%naspects = [6, 9];
%naspects = 9;

topic = 9;

alpha = 0.5;
%attr_num = [1, 2, 3];
%type_num = [1, 2]; % 1 : sum, 2 : prod

attr = 1;
type = 1;
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

Palpha = zeros(S, 9);
p = cell(S, 1);
pzs = cell(S, 1);

for s = 1 : S
    fprintf('session:%d\n', s);
    train_session = session;
    train_session(s) = [];
    test_session = session(s);
    for s2 = 1 : S - 1
        train_session(s2).prv = reg_attr(train_session(s2).content, train_session(s2).item, area_s(:, s)', alpha, attr, type);
    end
    test_session.prv = reg_attr(test_session.content, test_session.item, area_s(:, s)', alpha, attr, type);
    p{s} = train(train_session, topic, train_lim, train_num); %learn aspects with alpha:0.5
    pzs{s} = est(p{s}, test_session, est_lim); %estimate interests with alpha:0.5
    
    T = length(test_session.f);
    dt = T / 12;
    for t = 1 : 9
        f = test_session.f(floor((t-1)*dt+1) : ceil((t+3)*dt));
        Palpha(s, t) = est_alpha(f, p{s}.vz, pzs{s}, test_session.piv, area_s(:, s)');
    end
end


%%
%%% save
save(sprintf('./result/alpha_attr%d_type%d_alpha%d_likelihood_', attr, type, alpha * 100), 'Palpha', 'p', 'pzs');
