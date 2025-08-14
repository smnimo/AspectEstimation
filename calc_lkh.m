clear
%close all

%% load
load gazedata

%% param read
N_value = max(cell2mat(semantic_attributes(:)))';
N_att = size(cell2mat(semantic_attributes(:)), 2);
attrange = cell(1, N_att);
for n = 1 : N_att
    attrange{n} = 1 : N_value(n);
end
S = numel(semantic_attribute_seq);
lkh_ses = cell(S, 1);
for s = 1 : S
    lkh_ses{s} = cell(N_att, 1);
end

%% make all pair, max langth: 10
win_size = 5;
allpair3 = zeros(HomogenousProduct(3, win_size), 3); % N_value(1) ?????????????????????
m = 1;
for i = 0 : win_size
    for j = 0 : win_size - i
        k = win_size - i - j;
        allpair3(m, :) = [i, j, k];
        m = m + 1;
    end
end

%% loop for sessions
%s = 1;
for s = 1 : S
% calc probability of each attribute value (all region)
prob_att = cellfun(@(x, y) hist(x, y) ./ sum(hist(x, y)), num2cell(semantic_attributes{s}, [1 N_att]), attrange, 'UniformOutput', false);
attseq = semantic_attribute_seq{s};
T = size(attseq, 2);
%% ordinal variables
for att = 1 : 5 %ordinal cariables
%att = 1
valseq = attseq(att, :);
lkh_ses{s}{att} = ones(1, T);
m = size(allpair3, 1);
probset = zeros(m, 1);
for i = 1 : m
    probset(i) = exact_prob(prob_att{att}, allpair3(i, :));
end
%% calc likelihood
for t = win_size : T
    h = hist(valseq(t-win_size+1:t), 1:size(prob_att{1}, 2));
    lkh_ses{s}{att}(t) = probset(mean(allpair3==h, 2)==1);
end
end
end

%%save 
save('lkh_win_5', 'lkh_ses');

