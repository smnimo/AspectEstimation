%%% plot_duration_segmentation
close all
load gazedata
load segsessions
pathfunc

%%
dur = cell(3, 1);
S = numel(session);
for s = 1 : S
    dur{session{s}.session_id} = [dur{session{s}.session_id}; session{s}.totalduration];
end

figure('Name', 'hist of duration')
for trial = 1 : 3
subplot(3, 1, trial)
durhist = histogram(dur{trial}, 0 : 10 : 130);
%set(gca,'xticklabel',{'~10','~20','~30','~40','~50','~60','~70','~80','~90','~100','~110','~120','~130'})
title(sprintf('trial:%d, mean;%f, median:%f',trial, mean(dur{trial}), median(dur{trial})))
ylabel('number of sessions')
end
%% number ot gaze targets
itemnum = cell(3, 1);
S = numel(session);
for s = 1 : S
    itemnum{session{s}.session_id} = [itemnum{session{s}.session_id}; length(session{s}.f)];
end

figure('Name', 'hist of gaze targets')
for trial = 1 : 3
subplot(3, 1, trial)
itemhist = histogram(itemnum{trial}, 0 : 10: 150);
%bar(itemhist)
%xlim([0 17])
%set(gca,'xticklabel',{'~10','~20','~30','~40','~50','~60','~70','~80','~90','~100','~110','~120','~130', '~140', '~150', '~160', '~160', '~160', '~160', '~160', '~160'})
title(sprintf('trial:%d, mean:%f, median:%f', trial, mean(itemnum{trial}), median(itemnum{trial})))
ylabel('number of sessions')
end
%% number of gaze targets, which do not relate to given task
itemnum = cell(3, 1);
S = numel(session);
for s = 1 : S
    itemnum{session{s}.session_id} = [itemnum{session{s}.session_id}; sum(sum(session{s}.list(session{s}.f) ~= session{s}.candidate, 2)==2)];
end

figure('Name', 'hist of gaze targets')
for trial = 1 : 3
subplot(3, 1, trial)
itemhist = histogram(itemnum{trial}, 0 : 10: 100);
%bar(itemhist)
%xlim([0 17])
%set(gca,'xticklabel',{'~10','~20','~30','~40','~50','~60','~70','~80','~90','~100','~110','~120','~130', '~140', '~150', '~160', '~160', '~160', '~160', '~160', '~160'})
title(sprintf('trial:%d, mean:%f, median:%f', trial, mean(itemnum{trial}), median(itemnum{trial})))
ylabel('number of sessions')
end
%% ratio of focused gaze targets by gaze targets

focusedratio = cellfun(@(x) floor(nnz(x)/size(x, 2)*10)*10, seglabel);
ratiohist = hist(focusedratio, 0 : 10: 100);
figure('Name', sprintf('hist of focused ratio, mean:%f, median:%f', mean(focusedratio), median(focusedratio)))
bar(ratiohist)
xlim([0 12])
xlabel('the ratio focused targets');
ylabel('number of sessions')


%% the number of attribute values in attribute-of-focus

focusedvalue = cellfun(@sum, segsession, 'UniformOutput', false);
focusedvalue = cell2mat(focusedvalue');
focusedvalue = focusedvalue(focusedvalue>0);
focusedhist = normalize(hist(focusedvalue, 1 : 5))*100;
figure('Name', sprintf('hist of focused ratio, mean:%f, median:%f', mean(focusedvalue), median(focusedvalue)))
bar(focusedhist)
xlim([0 6])
xlabel('the number of focusing attribute values in attribute-of-focus');
ylabel('the number of gaze targets')


