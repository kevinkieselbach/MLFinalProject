function [ Ypred ] = evalClusterBoost(B, Xtest)
    Ypred = zeros(10,size(Xtest,2));
    for i = 1:size(B,2)
        label = evalCluster(B{i}.Cluster, Xtest);
        transformedLabels = zeros(10,size(Xtest,2));
        for j = 1:size(Xtest,2)
            transformedLabels(label(j), j) = 1;
        end
        Ypred = Ypred + B{i}.weight * transformedLabels;
    end
    [~,Ypred] = max(Ypred);
end