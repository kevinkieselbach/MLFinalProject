function [ B ] = adaboosttree(X, Y, maxIter, maxdepth)
    B = cell(0);
    n = size(X,2);
    weights = ones(1,n)/n;
    for i = 1:maxIter
        if rem(i, 10) == 0, disp(i), end
        T = id3tree(X,Y,maxdepth,weights);
        Ypred = evaltree(T,X);
        epsilon = sum(weights(Y ~= Ypred),2);
        if epsilon > 0.5, break; end
        if epsilon == 0,
            B{i} = struct('tree',T,'weight',0.5);
            break;
        end
        alpha = 0.5 * log((1-epsilon)/epsilon);
        B{i} = struct('tree',T,'weight',alpha);
        z = 2 * sqrt(epsilon*(1-epsilon));
        weights = weights .* exp(-alpha*(Ypred.*Y)) / z;
    end
end