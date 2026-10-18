function [ value ] = randchoose( range, prob, rate )

if(length(range) ~= length(prob))
	error 'not match size';
end
prob = normalize(prob);
N = length(prob);
box = [];
for i = 1 : N
	box = [box, ones(1, floor(prob(i)*rate))*i];
end
r = randperm(length(box));
value = range(box(r(1)));
