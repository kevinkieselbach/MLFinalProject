function [ B ] = randomForrest(X, Y)
    maxIter = 50;
    maxDepth = size(X,1);
    randFeatures = 20;
    
    B = cell(0);
    n = size(X,2);
    for i = 1:maxIter
        if rem(i, 10) == 0, disp(i), end
        indices = randsample(n,n);
        XSample = X(:,indices);
        YSample = Y(:,indices);
        T = id3tree(XSample,YSample,maxDepth , ones(1,n) /n, randFeatures);
        B{i} = struct('tree',T,'weight',1);
    end
    
end