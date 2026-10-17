clear
close all
load segsessions
load gazedata
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
L = size(seqmask{s}, 2)
for l = 1 : L
% number of AOF at time l (column l); the paper's Fig. 8 used seqmask{s}(l), a linear index into the
% 15 x L mask (elements of the first columns), see TODO.md A1
maskhist(floor((l-1) * 10 / L)+1) = maskhist(floor((l-1) * 10 / L)+1) + sum(seqmask{s}(:, l));
end
end
% B-3 (paper Fig. 8)
plot(maskhist)
ylabel('The total number of AOF')
xlabel('Temporal position in the session (10 segments)')
% B-4
figure
subplot(1, 2, 2)
boxplot(dc)
title('the number of items looked at in each task')
subplot(1, 2, 1)
boxplot(d)
title('the physical duration in each task')
