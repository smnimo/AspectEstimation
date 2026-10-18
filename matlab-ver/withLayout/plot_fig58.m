allalpha = [0 30 50 70 100];
h = './result/item/'
hoge = [];
topic = 6;
estimate_percent = [1 3 5 7 9];
for alpha = allalpha
    load(sprintf('%sMP_attr1_type1_alpha%d.mat', h, alpha));
    score = cell2mat(mpatk(topic, :));
    score = score(1, estimate_percent);
    hoge = [hoge; score];
end
bar(hoge');
