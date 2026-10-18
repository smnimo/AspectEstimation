clear all
close all

load ./result/alpha
load ./name
load ./session_id

S = length(content_id);
alpha = cell(9, 100);
col = [];
for s = 1 : S
    alpha{user_id(s), content_id(s) * 10 + task_id(s)} = Palpha(s, :);
    col = [col, content_id(s) * 10 + task_id(s)];
end
col = sort(col(1:9));
alpha = alpha(:, col)';

for s = 1 : S
    subplot(9, 9, s)
    plot(alpha{s})
end
    