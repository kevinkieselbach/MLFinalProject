function [ accuracy ] = PlattScaledSVM(X, Y, Xtest, Ytest)

    scaledSvmTestOutput = zeros(10,size(Xtest,2));
    for i = 1 : 10
       scaledSvmTestOutput(i,:) = PlattScaledSVMHelper(X, Y(i,:), Xtest); 
    end
    
    [~,Ypred] = max(scaledSvmTestOutput);
    [~,Ytest] = max(Ytest);
    
    correct = (Ypred == Ytest);
    accuracy = sum(correct(:))/size(Xtest,2);
end