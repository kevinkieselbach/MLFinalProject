function [ accuracy ] = Clustering( X, Y, XTest, YTest )
%CLUSTERING Summary of this function goes here
%   Detailed explanation goes here
centroids = zeros(size(X,1),10);
clusterAssignments = zeros(1,size(X,2));

bestCentroids = centroids;
bestClusterAssignments = clusterAssignments;
accuracy = 0;

for i = 1:100
  while (true)
      %take distance between all centroids for each point
      [~, newClusterAssignments] = min(distance(centroids, X));

      %check to make sure cluster assignments have changed
      if sum(newClusterAssignments ~= clusterAssignments,2) > 0
        clusterAssignments = newClusterAssignments;
        for j = 1:10
           %reassign position of centroids to means of the points under their
           %label
           centroids(:,j) = mean(X(:, clusterAssignments == j),2);
        end
      else
        % cluster assignments are the same so break, clustering finished
        break;
      end
  end
    
  % label centroids: mode of labels of points assigned to cluster
  centroidLabels = zeros(10, 1);
  for j = 1:10
     centroidLabels(j) = mode(Y(clusterAssignments == j));
  end
  
  % compute test error by comparing label of yTest to nearest cluster
  [~, testClusterAssignments] = min(distance(centroids, XTest));
  YPred = centroidLabels(testClusterAssignments);
  correct = (YTest == YPred);
  acc = sum(correct,2)/size(XTest,2);
  
  if (acc > accuracy)
    % bestCentroids = centroids;
    % bestClusterAssignments = clusterAssignments;
    accuracy = acc;
  end
  
end



