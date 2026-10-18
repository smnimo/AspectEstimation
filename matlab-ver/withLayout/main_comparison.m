clear all
close all

%%% load data
h = './data/mat/'; % place of dataset
session = struct;
load ./data/content.mat;
S = 81; % the number of sessions

%%% parameter setting
type_num = [1, 2, 3];
for type = type_num
    %%
    %%% load data
    fid = fopen('./data/experiment/timestamp.list');
    for s = 1 : S
        tline = fgetl(fid);
        load(strcat(h, tline, '.mat'));% load f and list, answer, content_id, tadk_id
        item = itemlist(3, 4, list);
        session(s).f = f;
        session(s).list = list;
        session(s).piv = item_attr(content, item, type);
        session(s).attr = makeattr(session(s).f, content, list, type);
        session(s).allattr = normalize(mean(session(s).attr));
        session(s).content_id = content_id;
        session(s).task_id = task_id;
        session(s).answer = zeros(length(list.index), 1);
        session(s).answer(answer) = 1;
    end
    fclose(fid);    
    
    %%
    %%% training except one task and evaluation
    %
    K = 3; % P@k
    patk_r = cell(1, 9);
    patk_i = cell(1, 9);
    mpatk_r = cell(1, 9);
    mpatk_i = cell(1, 9);
    for s = 1 : S
        fprintf('session: %d\n', s);
        train_session = session;
        train_session(s) = [];
        test_session = session(s);
        
        T = length(test_session.f);
        session_est = test_session;
        session_test = test_session;
        for ratio = 1 : 9
            session_est.f = test_session.f(1 : floor(ratio * T / 10));
            session_test.f = test_session.f(ceil(ratio * T / 10) : T);
            session_est.attr = makeattr(session_est.f, content, session_est.list, type);
            session_est.allattr = normalize(mean(session_est.attr));
            attr = est_cosine(train_session, session_est, 3);
            [patk_r{ratio}(:, s), patk_i{ratio}(:, s)] = eval_precision_comparison(session_test, content, attr, type, K);
        end
    end
    for ratio = 1 : 9
        mpatk_r{ratio} = mean(patk_r{ratio}, 2);
        mpatk_i{ratio} = mean(patk_i{ratio}, 2);
    end
    
    %%
    %%% save
    %
    mpatk = mpatk_r;
    save(sprintf('./result/region/MP_comp_type%d', type), 'mpatk');
    mpatk = mpatk_i;
    save(sprintf('./result/item/MP_comp_type%d', type), 'mpatk');    
end
