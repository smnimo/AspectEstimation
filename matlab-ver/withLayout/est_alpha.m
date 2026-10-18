function alpha = est_alpha(f, pvz, pzs, piv, layout)

hist_f = hist(f, 1 : 12);
pvs = pzs * pvz;
area = repmat(normalize(layout), 14, 1);

hoge = zeros(100001, 12);
r = 0;
for al = 0 : 0.00001 : 1
    r = r + 1;
    hoge(r, :) = pvs * ((1 - al) * piv + al * area);
end
%fuga = pdist2(hoge, hist_f, 'cosine');
fuga = log(hoge) * hist_f';
[~, mi] = max(fuga);
alpha = (mi - 1) * 0.00001;
