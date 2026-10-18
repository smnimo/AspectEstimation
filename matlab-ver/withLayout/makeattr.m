function attr = makeattr(f, content, list, type)


if type == 1
    attr = zeros(length(f), size(content.category{1}, 2) + size(content.calorie{1}, 2) + size(content.step{1}, 2));
    for i = 1 : length(f)
        attr(i, :) = [content.category{list.index(f(i))}(list.num(f(i)), :), content.calorie{list.index(f(i))}(list.num(f(i)), :), content.step{list.index(f(i))}(list.num(f(i)), :)];
    end
elseif type == 2
    attr = zeros(length(f), size(content.colhist{1}{1}, 2));
    for i = 1 : length(f)
        attr(i, :) = content.colhist{list.index(f(i))}{list.num(f(i))};
    end
else
    attr = zeros(length(f), size(content.category{1}, 2) + size(content.calorie{1}, 2) + size(content.step{1}, 2) + size(content.colhist{1}{1}, 2));
    for i = 1 : length(f)
        attr(i, :) = [content.category{list.index(f(i))}(list.num(f(i)), :), content.calorie{list.index(f(i))}(list.num(f(i)), :), content.step{list.index(f(i))}(list.num(f(i)), :), content.colhist{list.index(f(i))}{list.num(f(i))}];
    end
end
