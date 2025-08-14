%%% learn aspects --> plot transition of interests
function [ seglabel, seqmask ] = calc_segments(lkh_ses, semantic_attributes, semantic_attribute_seq, win_size, thr)

S = length(lkh_ses);
seglabel = cell(S, 1); % segment label'
seqmask = cell(S, 1);

N_value = max(semantic_attribute_seq{1}, [], 2);
N_att = size(semantic_attribute_seq{1}, 1);
attrange = cell(1, N_att);
for n = 1 : N_att
    attrange{n} = 1 : N_value(n); % attribute value range
end

%%
for s = 1 : S
    attseq = semantic_attribute_seq{s};
    prob_att = cellfun(@(x, y) hist(x, y) ./ sum(hist(x, y)), num2cell(semantic_attributes{s}, [1 N_att]), attrange, 'UniformOutput', false); % p_k
    T = size(lkh_ses{s}{1}, 2); % session length
    looked_att = cell(N_att, 1);
    
    for att = 1 : N_att
        valseq = attseq(att, :);
        looked_value = zeros(N_value(att), T);
        tmp = find(lkh_ses{s}{att}<thr);
        for i = 1 : length(tmp)
            x1 = tmp(i) - win_size+1;
            x2 = tmp(i);
            valflag =  (normalize(hist(valseq(x1:x2), 1:size(prob_att{1}, 2))) >= prob_att{att}); 
            for v = attrange{att}
                if valflag(v)
                    looked_value(v, x1:x2)=1;
                end
            end
        end
        looked_att{att} = looked_value;
    end
    %%
    looked_att = cell2mat(looked_att);
    sl = sum(looked_att);
    seglabel{s} = zeros(1, T); % segments ID
    if(sl(1)==0)
        seglabel{s}(1) = 0;
    else
        seglabel{s}(1) = 1;
    end
    segc = seglabel{s}(1);
    for t = 2 : T
        if sl(t) == 0
            seglabel{s}(t) = 0;
        elseif sum(abs(looked_att(:, t) - looked_att(:, t-1))) > 0
            segc = segc + 1;
            seglabel{s}(t) = segc;
        else
            seglabel{s}(t) = segc;
        end
    end
    seqmask{s} = looked_att;
end

