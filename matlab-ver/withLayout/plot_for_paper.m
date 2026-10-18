clear all
close all

name = {'item', 'region'};
fuga = cell(4, 3);

plot_range = [5];

r_k = [4, 10];
%topic6, mpat1
%topic9, mpat1
rodd = [1, 3, 5];
reven = [2, 4, 6];

for n = 1 : 2
    figure('name', sprintf('%s', name{n}));
end
for attr = 1 : 3
    for n = 1 : 2
        figure(n)
        hoge = [];
        
        load(sprintf('./result/%s/MP_attr%d_type1_alpha0.mat', name{n}, attr));
        mpatk = cell2mat(mpatk);
        mpatk = mpatk(r_k, :);
        
        hoge = [hoge; mpatk(:, plot_range)];
        
        alpha = 50;
        for type = 1 : 2
            load(sprintf('./result/%s/MP_attr%d_type%d_alpha%d.mat', name{n}, attr, type, alpha));
            mpatk = cell2mat(mpatk);
            mpatk = mpatk(r_k, :);
            hoge = [hoge; mpatk(:, plot_range)];
        end
        
        hoge = [hoge(rodd, :); hoge(reven, :)];
        
        load(sprintf('./result/%s/MP_comp_type%d.mat', name{n}, type));
        hoge = [hoge; mpatk{1, plot_range}];
        
        subplot(1, 3, attr)
        bar(hoge');
        ylim([0 0.7]);
        legend('topic6:org', 'topic6:sum', 'topic6:prod', 'topic9:org', 'topic9:sum', 'topic9:prod', 'comparison');
    end
end
