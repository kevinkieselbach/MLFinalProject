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
    
    % Clutering test
    [finalYPred, accuracy] = Clustering(Xtrain, Ytrain, Xtest, Ytest);
    display(accuracy)
    
    % Platt Scaled SVM
    accuracy = PlattScaledSVM(Xtrain, Ytrain, Xtest, Ytest);
    display(accuracy)
    
    % Adaboosted Decision Trees
    %accuracy = AdaboostedDecisionTrees(Xtrain, Ytrain, Xtest, Ytest);
    %display(accuracy)
end

