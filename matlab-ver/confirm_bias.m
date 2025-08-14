%%% to confirm exact prob, how long does it takes to observe bias
clear
pathfunc

%{
allpair2 = cell(10, 1);
for n = 1 : 10
    allpair2{n} = zeros(HomogenousProduct(2, n), 2);
    m = 1;
    for i = 0 : n
        j = n - i;
        allpair2{n}(m, :) = [i, j];
        m = m + 1;
    end
end

prob = normalize(ones(1, 2));
probset = cell(10, 1);
for n = 1 : 10
    m = size(allpair2{n}, 1);
    probset{n} = zeros(m, 1);
    for i = 1 : m
        probset{n}(i) = exact_prob(prob, allpair2{n}(i, :));
    end
end
pvalue = cell(10, 1);
for n = 1 : 10
    m = size(allpair2{n}, 1);
    pvalue{n} = zeros(m, 1);
    for i = 1 : m
        thr = probset{n}(i);
        pvalue{n}(i) = sum(probset{n}(probset{n}<=(thr+1e-16))); % sum Py (all Py <= exact_prob)
    end
end
%}


allpair3 = cell(10, 1);
for n = 1 : 10
    allpair3{n} = zeros(HomogenousProduct(3, n), 3); % N_value(1) ?????????????????????
    m = 1;
    for i = 0 : n
        for j = 0 : n - i
            k = n - i - j;
            allpair3{n}(m, :) = [i, j, k];
            m = m + 1;
        end
    end
end

prob = normalize(ones(1, 3));

probset = cell(10, 1);
for n = 1 : 10
    m = size(allpair3{n}, 1);
    probset{n} = zeros(m, 1);
    for i = 1 : m
        probset{n}(i) = exact_prob(prob, allpair3{n}(i, :));
    end
end

pvalue = cell(10, 1);
for n = 1 : 10
    m = size(allpair3{n}, 1);
    pvalue{n} = zeros(m, 1);
    for i = 1 : m
        thr = probset{n}(i);
        pvalue{n}(i) = sum(probset{n}(probset{n}<=(thr+1e-16))); % sum Py (all Py <= exact_prob)
    end
end


%{
allpair4 = cell(10, 1);
for n = 1 : 10
    allpair4{n} = zeros(HomogenousProduct(4, n), 4); % N_value(1) ?????????????????????
    m = 1;
    for i = 0 : n
        for j = 0 : n - i
            for k = 0 : n - i - j
                l = n - i - j - k;
                allpair4{n}(m, :) = [i, j, k, l];
                m = m + 1;
            end
        end
    end
end

prob = normalize(ones(1, 4));

probset = cell(10, 1);
for n = 1 : 10
    m = size(allpair4{n}, 1);
    probset{n} = zeros(m, 1);
    for i = 1 : m
        probset{n}(i) = exact_prob(prob, allpair4{n}(i, :));
    end
end

pvalue = cell(10, 1);
for n = 1 : 10
    m = size(allpair4{n}, 1);
    pvalue{n} = zeros(m, 1);
    for i = 1 : m
        thr = probset{n}(i);
        pvalue{n}(i) = sum(probset{n}(probset{n}<=(thr+1e-16))); % sum Py (all Py <= exact_prob)
    end
end
%}