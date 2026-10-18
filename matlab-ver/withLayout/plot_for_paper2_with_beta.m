clear all
close all

name = {'item', 'region'};
beta = {'', '_beta'};
fuga = cell(4, 3);

plot_range = [5];

r_k = [4];
%topic9, mpat1

attr = 1;
hoge = {[], []};
for n = 1 : 2
    
    load(sprintf('./result/%s/MP_attr%d_type1_alpha0.mat', name{n}, attr));
    mpatk = cell2mat(mpatk);
    mpatk = mpatk(r_k, :);
    
    hoge{n} = [hoge{n}; mpatk(:, plot_range)];
    
    alpha = 50;
    for type = 1 : 2
        for m = 1 : 2
            load(sprintf('./result/%s/MP_attr%d_type%d_alpha%d%s.mat', name{n}, attr, type, alpha, beta{m}));
            mpatk = cell2mat(mpatk);
            mpatk = mpatk(r_k, :);
            hoge{n} = [hoge{n}; mpatk(:, plot_range)];
        end
    end
        
end
bar(cell2mat(hoge)');
ylim([0 0.7]);
legend('org', 'sum', 'sum-beta', 'prod', 'prod-beta');
