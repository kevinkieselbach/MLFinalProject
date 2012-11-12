function MLFinalProjectMain()
    % Read in training data
    data = load('usps_resampled.mat');
    Xtrain = data.train_patterns;
    Ytrain = data.train_labels;
    
    Xvalidation = data.test_patterns(:, 1:2324);
    Yvalidation = data.test_labels(:, 1:2324);
    
    XfinalTest = data.test_patterns(:, 2325:end);
    YfinalTest = data.test_labels(:, 2325:end);
    
    Xtest = Xvalidation;
    Ytest = Yvalidation;
    
    % KNN test
    %accuracy = KNN(Xtrain, Ytrain, Xtest, Ytest);
    %display(accuracy)
    
    % Clustering test
<<<<<<< HEAD
%     accuracy = adaboostClusters(Xtrain, Ytrain, Xtest, Ytest);
=======
    %accuracy = adaboostClusters(Xtrain, Ytrain, Xtest, Ytest);
>>>>>>> 194fc9bb9600edbfbd8c6f1c6a7e52eb66171480
    
%     w = [ 1 2 3 4 5 6 7 8 9 10 ];
%     Ytest(Ytest < 0) = 0;
%     Ytest = w * Ytest;
%     correct = sum(Ytest == evalCluster(Clustering(Xtrain, Ytrain, ones(1,size(Xtrain,2))), Xtest),2);
%     accuracy = sum(correct(:))/size(Xtest,2);
    
<<<<<<< HEAD
%     display(accuracy)
=======
    %display(accuracy)
    
    % AdaboostedClustering
>>>>>>> 194fc9bb9600edbfbd8c6f1c6a7e52eb66171480
    
    % Platt Scaled SVM
    %accuracy = PlattScaledSVM(Xtrain, Ytrain, Xtest, Ytest);
    %display(accuracy)
    
    % Adaboosted Decision Trees
<<<<<<< HEAD
    accuracy = AdaboostedDecisionTrees(Xtrain, Ytrain, Xtest, Ytest);
=======
    %accuracy = AdaboostedDecisionTrees(Xtrain, Ytrain, Xtest, Ytest);
    %display(accuracy)
    
    %Adaboosted Decision Trees
    accuracy = baggedTrees(Xtrain, Ytrain, Xtest, Ytest);
>>>>>>> 194fc9bb9600edbfbd8c6f1c6a7e52eb66171480
    display(accuracy)
end

