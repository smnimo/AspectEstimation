%%% task-related attribute values (paper Table 1) -> task_att.mat
%%% feature index of attribute a and value v: (a - 1) * 3 + v
%%% attributes: price, screen size, CPU score, memory capacity, weight; values 1 = low/small/light .. 3 = high/large/heavy
clear
task_att = {[9, 12]; [4, 13]; [1, 12]}; % task 1: high CPU score, large memory capacity
                                        % task 2: small screen, light
                                        % task 3: low price, large memory capacity
giventask = zeros(numel(task_att), 15);
for k = 1 : numel(task_att)
    giventask(k, task_att{k}) = 1;
end
save('task_att', 'giventask', 'task_att');
