function [ piv ] = item_attr(content, item, attr)
% [prv, prvt] = reg_attr(content(o), item(o))
%
% content contains oll items, so num shows layouted items
% 
% prv       : probablistic of region R, conditioned by attribute value V, P(r|v) V_num x Region_num
% content   : object. item-attribute contents table
% item      : object. number of items layouted, and page
% rayout    : effect of rayout

for i = 1 : item.num
    if attr == 1
        %%% only descriptive
        pvr(i, :) = [content.category{item.list.index(i)}(item.list.num(i), :), content.calorie{item.list.index(i)}(item.list.num(i), :), content.step{item.list.index(i)}(item.list.num(i), :)];
    elseif attr == 2
        %%% only colorhistogram
        pvr(i, :) = content.colhist{item.list.index(i)}{item.list.num(i)};
    elseif attr ==3
        %%% descriptive and colorhistogram
        pvr(i, :) = [content.category{item.list.index(i)}(item.list.num(i), :), content.calorie{item.list.index(i)}(item.list.num(i), :), content.step{item.list.index(i)}(item.list.num(i), :), content.colhist{item.list.index(i)}{item.list.num(i)}];
    end
end
pvr(:, sum(pvr)==0) = 1;
piv = normalize(pvr', 2);

