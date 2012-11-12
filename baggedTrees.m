function [ accuracy ] = baggedTrees(X, Y, Xtest, Ytest)
    testOutput = zeros(10,size(Xtest,2));
    for i = 1 : 10
       B = randomForrest(X, Y(i,:));
       testOutput(i,:) = evalboost(B,Xtest);
       display(i)
    end
    
    [~,Ypred] = max(testOutput);
    [~,Ytest] = max(Ytest);
    
    correct = (Ypred == Ytest);
    accuracy = sum(correct(:))/size(Xtest,2);
end