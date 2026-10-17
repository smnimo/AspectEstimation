function plot_mask(sessions)
% plot_mask(sessions)
% attribute values of the browsed items (top) and detected attributes-of-focus (bottom) of each session
% sessions: linear indices of the session cells (column-major: (task - 1) * n_usr + usr); default 17 (paper Fig. 7)
% input: segsessions.mat (learn_aspects_from_mask.m), gazedata.mat
% (replaces disp_mask.m, disp_mask_for_paper.m and plot_mask_for_paper.m, which recomputed the masks;
%  plot_mask_for_paper.m showed the sessions [17, 34, 46])
if nargin < 1
    sessions = 17;
end
load('segsessions', 'session_all', 'seqmask');
load('gazedata', 'session', 'semantic_attributes');
N_value = max(cell2mat(semantic_attributes(:)))';
attributes = {'Price', 'Screen size', 'CPU score', 'Memory capacity', 'Weight'};
offset = [0; cumsum(N_value)];
centers = offset(1:end-1) + (0:numel(N_value)-1)' + (N_value + 1) / 2; % rows of the attributes incl. separators

for s = sessions
    figure('Name', sprintf('participant:%d, task:%d', session{s}.usr, session{s}.task))
    colormap gray
    subplot(2, 1, 1)
    imagesc(with_separators(session_all{s}, N_value), [0, 1])
    set(gca, 'YTick', centers, 'YTickLabel', attributes(1:numel(N_value)))
    title('(a) attribute values of browsed items')
    subplot(2, 1, 2)
    imagesc(with_separators(seqmask{s}, N_value), [0, 1])
    set(gca, 'YTick', centers, 'YTickLabel', attributes(1:numel(N_value)))
    title('(b) detected attributes-of-focus')
    xlabel('Time')
end
end

function out = with_separators(m, N_value)
% gray rows (0.5) between the attributes; values of an attribute from low (top) to high (bottom)
offset = [0; cumsum(N_value)];
out = [];
for a = 1 : numel(N_value)
    out = [out; m(offset(a)+1 : offset(a+1), :)];
    if a < numel(N_value)
        out = [out; 0.5 * ones(1, size(m, 2))];
    end
end
end
