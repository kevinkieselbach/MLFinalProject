function [ accuracy ] = adaboostClusters(X, Y, Xtest, Ytest)
    w = [ 1 2 3 4 5 6 7 8 9 10 ];
    YLabels = Y;
    YLabels(YLabels < 0) = 0;
    YLabels = w * YLabels;
    
    YtestLabels = Ytest;
    YtestLabels(YtestLabels < 0) = 0;
    YtestLabels = w * YtestLabels;
    
    B = cell(0);
    n = size(X,2);
    weights = ones(1,n)/n;
    
    for i = 1:1000
        Cluster = Clustering(X,Y,weights);
        Ypred = evalCluster(Cluster,X);
        epsilon = sum(weights(YLabels ~= Ypred),2)/sum(weights,2);
        if epsilon > 0.9, break; end
        if epsilon == 0,
            B{i} = struct('Cluster',Cluster,'weight',0.5);
            break;
        end
        alpha = log((1-epsilon)/epsilon) + log(9);
        B{i} = struct('Cluster',Cluster,'weight',alpha);
        weights = weights .* exp(alpha*(Ypred ~= YLabels)) / sum(weights,2);
        
        if rem(i, 10) == 0 
            disp(i);
            correct = sum(YtestLabels == evalClusterBoost(B,Xtest));
            accuracy = sum(correct(:))/size(Xtest,2)
        end
    end

%     for i = 1:100
%         if rem(i, 10) == 0, disp(i), end
%         Cluster = Clustering(X,Y,weights);
%         Ypred = evalCluster(Cluster,X);
%         
%         correct = sum(YtestLabels == evalCluster(Cluster,Xtest));
%         accuracy = sum(correct(:))/size(Xtest,2)
%         
%         epsilon = sum(weights(YLabels ~= Ypred),2);
%         display(epsilon)
%         if epsilon > 0.9,
%             display('broke here:');
%             display(i);
%             break;
%         end
%         if epsilon == 0,
%             B{i} = struct('Cluster',Cluster,'weight',0.5);
%             break;
%         end
%         alpha = 0.5 * log((1-epsilon)/epsilon);
%         B{i} = struct('Cluster',Cluster,'weight',alpha);
%         z = 2 * sqrt(epsilon*(1-epsilon));
%         weights = weights .* exp(-alpha*(Ypred ~= YLabels)) / z;
%     end
    
    correct = sum(YtestLabels == evalClusterBoost(B,Xtest));
    accuracy = sum(correct(:))/size(Xtest,2);
end