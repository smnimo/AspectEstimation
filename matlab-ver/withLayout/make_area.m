S = 81;
h = './data/mat/'; % place of dataset
fid = fopen('./data/experiment/timestamp.list');
a = cell(S, 1);
for s = 1 : S
	tline = fgetl(fid);
	load(strcat(h, tline, '.mat'));% load f and list, answer, content_id, tadk_id
    a{s} = hist(f, 1 : 12);
end
fclose(fid);

area_s = zeros(12, 81);
for t = 1 : S
    allS = 1 : S;
    allS(t) = [];
    area = [];
    for s = allS
        area = [area; a{s}];
    end
    area_s(:, t) = normalize(sum(area)');
end
%save('area_s.mat', 'area_s');
fuga = [];
for t = 1 : S
    allS = 1 : S;
    allS(t) = [];
    area = [];
    for s = allS
        area = [area; a{s}];
    end
    area_a = reshape(sum(area), 4, 3)';
    hoge = [];
    for i = 1 : 12
        x = mod(i - 1, 4) + 1;
        y = (i - x) / 4 + 1;
        hoge = [hoge; repmat([x, y], area_a(y, x), 1)];
    end  
    
    mu = [2.5, 1.5];
    hoge = hoge - repmat(mu, size(hoge, 1), 1);
    sig = hoge' * hoge / size(hoge, 1);
    fuga = [fuga diag(sig)]
end
%{
sig(1, 2) = 0;
sig(2, 1) = 0;

area_p = zeros(12, 1);
for i = 1 : 12
    x = mod(i - 1, 4) + 1;
    y = (i - x) / 4 + 1;
    area_p(i) = mvnpdf([x, y], mu, sig);
end
area = struct;
area.p = area_p;
area.mu = mu;
area.sig = sig;
%}
%save('area_p.mat', 'area');
area_s = zeros(12, 81);
for s = 1 : 81
%    mu = area.mu;
    sig = fuga(:, s)';
    for i = 1 : 12
        x = mod(i - 1, 4) + 1;
        y = (i - x) / 4 + 1;
        area_s(i, s) = mvnpdf([x, y], mu, sig);
    end
end
sig_s = fuga;
%save('sig.mat', 'area_s', 'sig_s');
save('sig_.mat', 'area_s', 'sig_s');
