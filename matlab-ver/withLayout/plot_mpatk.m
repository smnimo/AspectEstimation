clear all
close all

name = cell(2, 1);
name{1} = 'item';
name{2} = 'region';
fuga = cell(4, 1);

%beta_on = 1;
beta = {'_beta', ''};

for n = 1 : 2    
    attr = 1;
    
    figure('name', sprintf('%s', name{n}));
    hoge = cell(4, 1);
    
    type = 1;
    alpha_num = [0];
    
    for alpha = alpha_num
        load(sprintf('./result/%s/MP_attr%d_type%d_alpha%d.mat', name{n}, attr, type, alpha));
        mpatk = cell2mat(mpatk);
        [r, c] = size(mpatk);
        r_k = (mod([1 : r],  3) == 1);
        mpatk = mpatk(r_k, :);
        
        for i = 1 : 4
            hoge{i} = [hoge{i}; mpatk(i, :)];
        end
    end
    
    for beta_on = 2 : -1 : 1
    alpha = 50;    
    for type = 1 : 2
        load(sprintf('./result/%s/MP_attr%d_type%d_alpha%d%s.mat', name{n}, attr, type, alpha, beta{beta_on}));
        mpatk = cell2mat(mpatk);
        mpatk = mpatk(r_k, :);
        for i = 1 : 4
            hoge{i} = [hoge{i}; mpatk(i, :)];
        end        
    end
    end
    
    for i = 1 : 4
        subplot(4, 1, i)
        bar(hoge{i}');
        ylim([0 0.6]);
        title(sprintf('topic:%d', i * 3));
        fuga{i} = [fuga{i} mean(hoge{i}, 2)];
    end
    legend('alpha0', 'sum' , 'prod', 'beta-sum', 'beta-prod');
end

figure
for i = 1 : 4
    subplot(4, 1, i);
    bar(fuga{i}');
    ylim([0 0.6]);
    title(sprintf('topic:%d', i * 3));
end
legend('alpha0', 'sum', 'prod', 'beta-sum', 'beta-prod');
