function [alpha, b] = trainSVM(Kernel, trainLabels, C)

    % Determine the number of training points.
    [ numTrainPoints N ] = size(Kernel);
    if numTrainPoints ~= N
        error('Kernel must be a square matrix of size numTrainPoints by numTrainPoints\n');
    end
    if numTrainPoints ~= size(trainLabels,1)
       display(numTrainPoints)
       display(size(trainLabels))
       error('The number of training labels must match the number of trainining points\n'); 
    end
    
    % In this algorithm, lambda is the weight for the regularization term,
    % not the cost term. This is the opposite of how C was used in the
    % class notes. To correct for this difference, I set lambda to the
    % reciprocal of C.
    lambda = 1/C;
    
    % Use a sample of k training points on each iteration.
    k = numTrainPoints;
    
    % Initialize alpha to all zeros (i.e. initialize w to be the zero
    % vector).
    alpha = zeros(numTrainPoints,1);
    
    % Initialize b to zero.
    b = 0;
    
    % Repeat until convergence.
    for t = 1:100
        % Choose a subset of the training points.
        sampleIndices = randsample(numTrainPoints, k);
        
        % Define a function that computes the inner product between w and
        % xi for each index passed as a parameter. The return value has a
        % row for each index.
        wxiInnerProduct = @(alphaCopy,indices) sum( repmat(alphaCopy,1,max(size(indices))) .* Kernel(:,indices) )';
        
        % Determine the sample indices where there is a non-zero loss (i.e.
        % y times the inner product of w and xi is less than 1).
        sampleIndicesWithLoss = sampleIndices( trainLabels(sampleIndices) .* (wxiInnerProduct(alpha,sampleIndices) + b) < 1 );
        
        % Compute the step size, which decreases on each iteration.
        eta = 1/(lambda*t);
        
        % Update alpha to take a step in the direction of the gradient.
        alpha = (1 - eta*lambda) * alpha;
        alpha(sampleIndicesWithLoss) = alpha(sampleIndicesWithLoss) + eta/k*trainLabels(sampleIndicesWithLoss);
        
        % Update b to take a step in the direction of the gradient.
        %b = b + eta/k*sum(trainLabels(sampleIndicesWithLoss));
        
        % Define a function that computes the L2 norm of w.
        wL2Norm = @(alphaCopy) sum( alphaCopy .* wxiInnerProduct(alphaCopy,1:numTrainPoints) );
        
        % Project w onto the set of vectors of magnitude less than or equal
        % to 1 divided by the square root of lambda.
        alpha = min(1, 1/sqrt(lambda)/wL2Norm(alpha)) * alpha;
        
        if rem(t,10) == 0, disp(t), end
    end
end