function [ out ] = bhattacharyya(a, b)
if min(size(a)) > 1 || min(size(b)) > 1
    fprintf('error\n')
    return
end
if length(a) ~= length(b)
    fprintf('error\n')
    return
end
f = length(a);
out = zeros(f, 1);
for i = 1 : f
    out(i) = sqrt(a(i) * b(i));
end
out = -1 * log(sum(out));
end

