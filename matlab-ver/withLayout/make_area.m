%%% layout effect P(R; phi) for the leave-one-out evaluation (paper Eq. 11): relative gaze frequency of each region
%%% over all sessions except the evaluated one
%%% input: data/mat/*.mat, data/experiment/timestamp.list; output: area_s.mat (area_s: 12 regions x 81 sessions)
S = 81;
h = './data/mat/'; % place of dataset
fid = fopen('./data/experiment/timestamp.list');
a = zeros(S, 12);
for s = 1 : S
	tline = fgetl(fid);
	load(strcat(h, tline, '.mat'));% load f and list, answer, content_id, tadk_id
    a(s, :) = hist(f(:, 1)', 1 : 12);
end
fclose(fid);

area_s = zeros(12, S);
for t = 1 : S
    allS = 1 : S;
    allS(t) = [];
    area_s(:, t) = normalize(sum(a(allS, :), 1)');
end
save('area_s.mat', 'area_s');
