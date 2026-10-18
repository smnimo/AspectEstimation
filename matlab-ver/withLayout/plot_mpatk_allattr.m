clear all
close all

name = cell(2, 1);
name = {'item', 'region'};
beta = {'_beta', ''};
beta_on = 2;

fuga = cell(4, 3);

plot_range = [5, 7];

for n = 1 : 2
    figure('name', sprintf('%s', name{n}));
end    
for attr = 1 : 3
for n = 1 : 2    
%    attr = 1;
    
%    figure('name', sprintf('%s', name{n}));
    figure(n)
    hoge = cell(4, 1);
    
    type = 1;
    alpha_num = [0];
    
    for alpha = alpha_num
        load(sprintf('./result/%s/MP_attr%d_type%d_alpha%d%s.mat', name{n}, attr, type, alpha, beta{beta_on}));
        mpatk = cell2mat(mpatk);
        [r, c] = size(mpatk);
        r_k = (mod([1 : r],  3) == 1);
        mpatk = mpatk(r_k, :);
        
        for i = 1 : 4
%            hoge{i} = [hoge{i}; mpatk(i, :)];
            hoge{i} = [hoge{i}; mpatk(i, plot_range)];
        end
    end
    
    alpha = 50;    
    for type = 1 : 2
        load(sprintf('./result/%s/MP_attr%d_type%d_alpha%d%s.mat', name{n}, attr, type, alpha, beta{beta_on}));
        mpatk = cell2mat(mpatk);
        mpatk = mpatk(r_k, :);
        for i = 1 : 4
%            hoge{i} = [hoge{i}; mpatk(i, :)];
            hoge{i} = [hoge{i}; mpatk(i, plot_range)];
        end        
    end

        load(sprintf('./result/%s/MP_comp_type%d.mat', name{n}, type));
        for i = 1 : 4
%            hoge{i} = [hoge{i}; mpatk(i, :)];
            hoge{i} = [hoge{i}; mpatk{1, plot_range}];
        end        
    
    
    
    for i = 1 : 4
        subplot(4, 3, (i-1)*3 + attr)
        bar(hoge{i}');
        ylim([0 0.7]);
        title(sprintf('topic:%d', i * 3));
        fuga{i, attr} = [fuga{i, attr} mean(hoge{i}, 2)];
    end
    legend('alpha0', 'sum', 'prod');
end
end
figure('name', sprintf('%d', plot_range));
for attr = 1 : 3
for i = 1 : 4
    subplot(4, 3, (i-1)*3 + attr);
    bar(fuga{i, attr}');
    ylim([0 0.6]);
    title(sprintf('topic:%d', i * 3));
end
end
legend('alpha0', 'sum', 'prod', 'comparison');
