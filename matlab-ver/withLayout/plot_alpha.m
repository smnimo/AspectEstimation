function [] = plot_alpha(pass, type, naspect)

attr = 1;
head = sprintf('MP_attr%d_type%d_alpha', attr, type);
alpha_num = [0, 30, 50, 70, 100];

tmp = cell(length(alpha_num), 1);
count = 1;
for alpha = alpha_num
%    tmp{count} = load(sprintf('%s%s%d_beta.mat', pass, head, alpha));
    tmp{count} = load(sprintf('%s%s%d.mat', pass, head, alpha));
    count = count + 1;
end

result = cell(9, 1);
for i = 1 : length(alpha_num)
    for j = 1 : 2 : 9
        result{j} = [result{j},tmp{i}.mpatk{naspect, j}(1)];
    end
end
figure
bar(cell2mat(result));
legend('0', '0.3', '0.5', '0.7', '1');
figure
plot(cell2mat(result)', 'o-');
legend('10', '30', '50', '70', '90');

