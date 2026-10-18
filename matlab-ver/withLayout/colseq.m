function [seq, col] = colseq(f, list, colhist)
% [ seq, col ] = colseq(f, colhist)
% seq     : sequence of gaze item
% col     : sequence of color histogram of gaze item
% f       : sequence of gaze region
% colhist : colorhistogram of contents

seq = f(:, 1);
region = f(:, 2 : size(f, 2));

item_colhist = zeros(length(list.index), size(colhist{1}{1}, 2));
for i = 1 : length(list.index)
    if size(colhist{list.index(i)}{list.num(i)}, 1) == 1
        item_colhist(i, :) = colhist{list.index(i)}{list.num(i)};
    else
        item_colhist(i, :) = sum(colhist{list.index(i)}{list.num(i)});
    end
end
all_colhist = sum(item_colhist);

col = zeros(size(seq, 1), size(colhist{1}{1}, 2));
for i = 1 : size(f, 1)
    con = list.index(seq(i));
    num = list.num(seq(i));
    region_id = find(region(i, :)>0);
    for j = 1 : length(region_id)
        col_id = find(colhist{con}{num}(region_id(j), :) > 0);
        col(i, :) = col(i, :) + colhist{con}{num}(region_id(j), :) * region(i, region_id(j)) * item_colhist(seq(i), col_id) / (all_colhist(col_id) +  (all_colhist(col_id) == 0));
    end
%    col_id = find(colhist{con}{num}(region, :)>0);
%    col(i, :) = colhist{con}{num}(region, :) * item_colhist(f(i, 1), col_id) / (all_colhist(col_id)+(all_colhist(col_id)==0));
end

    