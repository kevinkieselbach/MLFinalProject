function [ accuracy ] = AdaboostedDecisionTrees(X, Y, Xtest, Ytest)
<<<<<<< HEAD
    maxiter = 350;
    maxdepth = 5;
=======
    maxiter = 150;
    maxdepth = 2;
>>>>>>> 194fc9bb9600edbfbd8c6f1c6a7e52eb66171480

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