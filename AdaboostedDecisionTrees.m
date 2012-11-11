function [ accuracy ] = AdaboostedDecisionTrees(X, Y, Xtest, Ytest)
    maxiter = 150;
    maxdepth = 5;

    testOutput = zeros(10,size(Xtest,2));
    for i = 1 : 10
       B = adaboosttree(X, Y(i,:), maxiter, maxdepth);
       testOutput(i,:) = evalboost(B,Xtest);
    end
    
    [~,Ypred] = max(testOutput);
    [~,Ytest] = max(Ytest);
    
    correct = (Ypred == Ytest);
    accuracy = sum(correct(:))/size(Xtest,2);
end