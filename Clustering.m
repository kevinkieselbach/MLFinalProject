function [ accuracy ] = Clustering( X, Y, XTest, YTest )
%CLUSTERING Summary of this function goes here
%   Detailed explanation goes here

clusterAssignments = zeros(1,size(X,2));
numCentroids = 50
numIterations = 20
doPrint = false;
%change Y and YTest to represent single digit value rather than binary
%class in 10 dim
w = [ 1 2 3 4 5 6 7 8 9 10];
Y(Y < 0) = 0;
Y = w * Y;
YTest(YTest < 0) = 0;
YTest = w * YTest;

%bestCentroids = centroids;
%bestClusterAssignments = clusterAssignments;

accuracy = 0;

for i = 1:numIterations
  centroids = 2 * rand(size(X,1), numCentroids) - 1;
  while (true)
      %take distance between all centroids for each point
      [~, newClusterAssignments] = min(distance(centroids, X));

      %check to make sure cluster assignments have changed
      if sum(newClusterAssignments ~= clusterAssignments,2) > 0
        clusterAssignments = newClusterAssignments;
        for j = 1:numCentroids
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
  centroidLabels = zeros(1,numCentroids);
  for j = 1:numCentroids
    centroidLabels(j) = mode(Y(clusterAssignments == j));
  end  
  
  % compute test error by comparing label of yTest to nearest cluster
  [~, testClusterAssignments] = min(distance(centroids, XTest));
  YPred = centroidLabels(testClusterAssignments);

  correct = (YTest == YPred);
  testAccuracy = sum(correct,2)/size(XTest,2);
  
  % compute training error by comparing label of Y to nearest cluster
  if doPrint
    [~, testClusterAssignments] = min(distance(centroids, X));
    YPred = centroidLabels(testClusterAssignments);

    correct = (Y == YPred);
    trainAccuracy = sum(correct,2)/size(X,2)
  end
  
  if (testAccuracy > accuracy)
    % bestCentroids = centroids;
    % bestClusterAssignments = clusterAssignments;
    accuracy = testAccuracy; 
  end
  
  if mod(i,2) == 0 && doPrint
    testaccuracy = accuracy;
    display(testaccuracy);  
  end
  
end



