function target_dur = LoadGaze(data)

timestamp = data(:, 1);
gaze = data(:, 2:end);

x = gaze(:, [1,3]);
y = gaze(:, [2,4]);
x(x(:, 1)==-1, 1) = x(x(:, 1)==-1, 2);
x(x(:, 1)==-1, 2) = x(x(:, 1)==-1, 1);
x = mean(x, 2);

y(y(:, 1)==-1, 1) = y(y(:, 1)==-1, 2);
y(y(:, 1)==-1, 2) = y(y(:, 1)==-1, 1);
y = mean(y, 2);

gaze_seq = [x, y];
vN = 3;
cN = 4;
target_seq = floor(y * vN) * cN + floor(x * cN) + 1;
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

