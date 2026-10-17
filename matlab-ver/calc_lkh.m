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

%% window size
win_size = 5;

%% loop for sessions
%s = 1;
for s = 1 : S
% calc probability of each attribute value (all region)
prob_att = cellfun(@(x, y) hist(x, y) ./ sum(hist(x, y)), num2cell(semantic_attributes{s}, [1 N_att]), attrange, 'UniformOutput', false);
attseq = semantic_attribute_seq{s};
T = size(attseq, 2);
%% ordinal variables
for att = 1 : N_att
valseq = attseq(att, :);
lkh_ses{s}{att} = ones(1, T); % the first win_size-1 time points keep likelihood 1
%% calc likelihood (multinomial, paper Eq. 1; any number of values per attribute)
for t = win_size : T
    h = hist(valseq(t-win_size+1:t), attrange{att});
    lkh_ses{s}{att}(t) = exact_prob(prob_att{att}, h);
end
end
end

%%save 
save('lkh_win_5', 'lkh_ses');

