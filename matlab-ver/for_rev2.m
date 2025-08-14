clear
close all
load segsessions
load('/Users/simonisi/Dropbox/MATLAB/201909_Gaze/results/cor.mat')
d = zeros(37, 3)
for i = 1 : 111
d(i) = session{i}.totalduration
end
dc = zeros(37, 3)
for i = 1 : 111
dc(i) = length(session{i}.f)
end
dci = zeros(37, 3)
for i = 1 : 111
dci(i) = length(session{i}.f)/session{i}.totalduration;
end
maskhist = zeros(10, 1)
for s = 1 : 111
L = length(sum(seqmask{s}))
for l = 1 : L
maskhist(floor((l-1) * 10 / L)+1) = maskhist(floor((l-1) * 10 / L)+1) + seqmask{s}(l);
end
end
% B-3
plot(maskhist)
ylim([0, 50])
% B-4
figure
subplot(1, 2, 2)
boxplot(dc)
title('the number of items looked at in each task')
subplot(1, 2, 1)
boxplot(d)
title('the physical duration in each task')
