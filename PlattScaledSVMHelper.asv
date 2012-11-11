function [ scaledSvmTestOutput ] = PlattScaledSVMHelper(X, y, Xtest)
    n = size(X,2); % numTrainingPoints
    y01 = y;
    y01(y01<0) = 0;

    % Learn an SVM on the training set
    %[~,~,fun] = svm_kernel('X', X', 'Y', y', 'sigma', 1000, 'lambda', 1000);
    %[~,~,fun] = svm_kernel('X', X', 'Y', y', 'kernel', 'polynomial', 'degree', 3, 'lambda', 1);
    
    svmStruct = svmtrain(X', y', 'kernel_function', 'polynomial', 'polyorder', 3);
    fun = @(X) svmStruct.KernelFunction(X,svmStruct.SupportVectors) * svmStruct.Alpha + svmStruct.Bias;
    
    svmTrainOutput = [ ones(1,n) ; fun(X')' ];
    
    % Define the logistic function
    h_theta = @(theta, X) 1./(1 + exp(-theta'*X));
    
    % Define the cost function for logistic regression
    function [ cost, grad ] = costFunction(theta)
        cost = sum(1/n * log(1 + exp(-y .*(theta'*svmTrainOutput))), 2);
        grad = 1/n * svmTrainOutput * (h_theta(theta,svmTrainOutput) - y01)';
    end

    % Learn the parameters (theta) of the logistic function
    % using logistic regression. The input to the logistic
    % regression is a set of 2 dimensional vectors where
    % each vector is a constant term (=1) and the output
    % of the svm for a training point.
    theta = fminunc(@costFunction, zeros(2,1), optimset('GradObj','on'));
    
    % Pass the test points to the learned SVM and
    % Platt Scale the results using the learned 
    % logistic function parameters.
    svmTestOutput = [ ones(1,size(Xtest,2)) ; fun(Xtest')' ];
    scaledSvmTestOutput = h_theta(theta, svmTestOutput);
end