load('gazedata.mat')
S = numel(session);
dur = zeros(size(session, 1), 3);
for s = 1 : S
%dur(session{s}.usr, session{s}.session_id) = length(session{s}.f);
dur(session{s}.usr, session{s}.session_id) = session{s}.totalduration;
end

usr_nocomp2 = [28, 36];
usr_nocomp1 = [1, 3, 6, 18, 23, 35, 37];

subplot(2, 1, 1)
[~, id] = sort(mean(dur'));
boxplot(dur(id, :)', num2str(id'))
ylim([0 160])
nocomp2id = find(sum(id==usr_nocomp2'));
for s = 1 : length(usr_nocomp2)
    x = [nocomp2id(s)-0.5, nocomp2id(s)+0.5, nocomp2id(s)+0.5, nocomp2id(s)-0.5];
    y = [0, 0, 160, 160];
    patch(x, y, 'r')
end
nocomp1id = find(sum(id==usr_nocomp1'));
for s = 1 : length(usr_nocomp1)
    x = [nocomp1id(s)-0.5, nocomp1id(s)+0.5, nocomp1id(s)+0.5, nocomp1id(s)-0.5];
    y = [0, 0, 160, 160];
    patch(x, y, 'y')
end
alpha(0.3)
ylabel('duration(s)')
xlabel('participant id')
title('sort by mean')

subplot(2, 1, 2)
[~, id] = sort(median(dur'));
boxplot(dur(id, :)', num2str(id'))
ylim([0 160])
nocomp2id = find(sum(id==usr_nocomp2'));
for s = 1 : length(usr_nocomp2)
    x = [nocomp2id(s)-0.5, nocomp2id(s)+0.5, nocomp2id(s)+0.5, nocomp2id(s)-0.5];
    y = [0, 0, 160, 160];
    patch(x, y, 'r')
end
nocomp1id = find(sum(id==usr_nocomp1'));
for s = 1 : length(usr_nocomp1)
    x = [nocomp1id(s)-0.5, nocomp1id(s)+0.5, nocomp1id(s)+0.5, nocomp1id(s)-0.5];
    y = [0, 0, 160, 160];
    patch(x, y, 'y')
end
alpha(0.3)
ylabel('duration(s)')
xlabel('participant id')
title('sort by median')