function [ prv ] = reg_attr(content, item, layout, alpha, attr)
% prv = reg_attr(content, item, layout, alpha, attr)
%
% prv     : P(R_n | V_q) = (1 - alpha) P(I_n | V_q) + alpha P(R_n; phi) (paper Eq. 4), V_num x Region_num
% content : item attributes of all contents (data/content.mat)
% item    : itemlist object: displayed items and their layout
% layout  : effect of the layout P(R; phi), 1 x Region_num (relative gaze frequency, make_area.m)
% alpha   : mixture ratio
% attr    : attribute type, 1: semantic (category, calorie, steps), 2: color, 3: both

for i = 1 : item.num
    if attr == 1
        %%% only descriptive
        pvi(i, :) = [content.category{item.list.index(i)}(item.list.num(i), :), content.calorie{item.list.index(i)}(item.list.num(i), :), content.step{item.list.index(i)}(item.list.num(i), :)];
    elseif attr == 2
        %%% only colorhistogram
        pvi(i, :) = content.colhist{item.list.index(i)}{item.list.num(i)};
    elseif attr ==3
        %%% descriptive and colorhistogram
        pvi(i, :) = [content.category{item.list.index(i)}(item.list.num(i), :), content.calorie{item.list.index(i)}(item.list.num(i), :), content.step{item.list.index(i)}(item.list.num(i), :), content.colhist{item.list.index(i)}{item.list.num(i)}];
    end
end

effect = repmat(normalize(layout), [size(pvi, 2), 1]);
pvi(:, sum(pvi)==0) = 1; % a value that no displayed item has: uniform over the items
prv = normalize((1 - alpha) * normalize(pvi', 2) + alpha * effect, 2);
