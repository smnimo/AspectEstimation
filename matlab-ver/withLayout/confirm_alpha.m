%%% ??????????????????
clear all
close all

%%% load data
load ./result/alpha
load ./name

h = './data/mat/'; % place of dataset
session = struct;
load ./area_s.mat;
load ./data/content.mat;
S = 81; % the number of sessions
attr = 1;

fid = fopen('./data/experiment/timestamp.list');
for s = 1 : S
	tline = fgetl(fid);
	load(strcat(h, tline, '.mat'));% load f and list, answer, content_id, tadk_id
 	item = itemlist(3, 4, list);
 	[session(s).f, session(s).col] = colseq(f, item.list, content.colhist);
    session(s).piv = item_attr(content, item, attr);
end
fclose(fid);

%%
%%% training except one task and evaluation


for s = 1 : S
    fprintf('session:%d\n', s);
    figure('Name', name{s});
    for i = 1 : 9
        subplot(3, 9, i);
        imagesc(reshape(area_s(:, s), 4, 3)');
    end
    T = length(session(s).f);
    dt = T / 12;
    for t = 1 : 9
        f = session(s).f(floor((t-1)*dt+1) : ceil((t+3)*dt));
        subplot(3, 9, t + 9)
        imagesc(reshape(hist(f, 1:12), 4, 3)');
    end
    for i = 1 : 9
        subplot(3, 9, i + 18);
        imagesc(reshape(pzs{s} * p{s}.vz * session(s).piv, 4, 3)');
    end
    colormap gray
end
