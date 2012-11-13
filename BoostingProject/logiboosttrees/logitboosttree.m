function [ B ] = logitboosttree(X, Y, maxIter, maxdepth)
    B = cell(0);
    n = size(X,2);
    prob = ones(1,n)/2;
    Y(Y < 0) = 0;
    for i = 1:maxIter
        display(prob)
        weights = prob.*(1-prob);
        Z = (Y - prob)./weights;
        T = logitree(X,Z,maxdepth,weights);
        B{i} = struct('tree',T,'weight',0.5);
        evalB = evalboost(B,X);
        prob = exp(evalB) ./ (exp(evalB) + exp(-evalB));
        if sum(prob == 1) > 0, 
            display(i)
            break; 
        end
    end
    for i = 1:size(B,2)
        B{i}.weight = 1;
    end
end