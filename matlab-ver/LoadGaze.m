function target_dur = LoadGaze(data)

timestamp = data(:, 1);
gaze = data(:, 2:end);

x = gaze(:, [1,3]);
y = gaze(:, [2,4]);
% a missing eye (-1) is filled from the other eye (columns: right, left);
% the mask of column 2 is taken before filling column 1 (a missing left eye used to stay -1
% and the sample was dropped as (x_R - 1) / 2 < 0)
missing = x == -1;
x(missing(:, 1), 1) = x(missing(:, 1), 2);
x(missing(:, 2), 2) = x(missing(:, 2), 1);
x = mean(x, 2);

missing = y == -1;
y(missing(:, 1), 1) = y(missing(:, 1), 2);
y(missing(:, 2), 2) = y(missing(:, 2), 1);
y = mean(y, 2);

gaze_seq = [x, y];
vN = 3;
cN = 4;
% x == 1 / y == 1 are valid and belong to the last column / row
target_seq = min(floor(y * vN), vN - 1) * cN + min(floor(x * cN), cN - 1) + 1;
target_seq(x<0 | x>1 | y < 0 | y > 1) = -1;
%%
timestamp = timestamp - timestamp(1);
timestamp = timestamp / 1000 / 100; % 100ms
timestamp(target_seq<0) = [];
target_seq(target_seq<0) = [];

target_tb_te = [target_seq(diff([0; target_seq])~=0), timestamp(diff([0; target_seq])~=0), timestamp(diff([target_seq; 0])~=0)];
target_dur = [target_tb_te(:, 1), target_tb_te(:, 3) - target_tb_te(:, 2)];
thr = 1.5; %/100ms => 150ms
target_tb_te(target_dur(:, 2)<1.5, :) = [];

T = size(target_tb_te, 1);
t = 2;
while t <= T
    if target_tb_te(t, 1) == target_tb_te(t-1, 1)
        target_tb_te(t-1, 3) = target_tb_te(t, 3);
        target_tb_te(t, :) = [];
        T = T - 1;
    else
        t = t + 1;
    end
end
target_dur = [target_tb_te(:, 1), target_tb_te(:, 3) - target_tb_te(:, 2)];

