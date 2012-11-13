function [ accuracy ] = KNN(X, Y, Xtest, Ytest)
    k = 1; % k as in k nearest neighbors
    pcaDim = 100;
    
    [ U , ~, ~ ] = svd(X);
    U = U(:, 1:pcaDim);
    
    X = U' * X;
    Xtest = U' * Xtest;
    
    w = [ 0 1 2 3 4 5 6 7 8 9 ];
    Y(Y < 0) = 0;
    Y = w * Y;
    Ytest(Ytest < 0) = 0;
    Ytest = w * Ytest;
    
    numTrainingPoints = size(X, 2);
    numTestPoints = size(Xtest, 2);
    Ypred = zeros(1, numTestPoints);
    
    % Determine the label for each test point
    for ix = 1 : numTestPoints
        
        if rem(ix, 1000) == 0, disp(ix), end
        
        % Determine the k nearest neighbors for the test point
        kNN = Inf(2,k); % Each NN is stored as [distance; Y]
        for jx = 1 : numTrainingPoints
            
            % Compute the distance to a given neighbor and
            % update kNN if the neighbor is closer than one
            % of the neighbors in kNN
            dist = Dist(Xtest(:,ix), X(:,jx));
            for kx = 1 : size(kNN, 2)
                
                % Replace furthest away neighbor in kNN
                % with the new, closer neighbor
                if dist < kNN(1,kx)
                    kNN(:, kx+1:end) = kNN(:, kx:end-1);
                    kNN(:, kx) = [dist; Y(jx)];
                    break
                end
            end
        end
        % Assign the most common label among the 
        % k nearest neighbors to the test point
        Ypred(ix) = mode(kNN(2,:));
    end
    
    correct = (Ypred == Ytest);
    accuracy = sum(correct(:))/size(Xtest,2);
end 