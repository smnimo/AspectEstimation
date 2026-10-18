function [ prv ] = reg_attr(content, item, layout, alpha, attr, type)
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
if type == 1 % sum
    pvi(:, sum(pvi)==0) = 1;
    prv = normalize((1 - alpha) * normalize(pvi', 2) + alpha * effect, 2);
else % prod
    prv = normalize(pvi', 1);
    prv(sum(prv, 2)==0, :) = 1;
    prv = normalize(prv .* effect, 2);
end
