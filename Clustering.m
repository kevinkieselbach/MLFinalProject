function [ Cluster ] = Clustering( X, Y, varargin )
%CLUSTERING Summary of this function goes here
%   Detailed explanation goes here

%If there are more than 2 arguments, use the first optional argument
%as the weight vector and sample accordingly
if nargin > 2
    indices = randsample(size(X,2),size(X,2),true,varargin{1})';
    X = X(:,indices);
    Y = Y(:,indices);
end

clusterAssignments = zeros(1,size(X,2));
numCentroids = 150;
numIterations = 10;

%change Y and YTest to represent single digit value rather than binary
%class in 10 dim
w = [ 1 2 3 4 5 6 7 8 9 10 ];
Y(Y < 0) = 0;
Y = w * Y;

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
  Clustertemp = struct('centroids',centroids,'centroidLabels',centroidLabels);
  YPred = evalCluster(Clustertemp,X);

  correct = (Y == YPred);
  tempAccuracy = sum(correct,2)/size(X,2);
  
  if (tempAccuracy > accuracy)
    Cluster = Clustertemp;
    accuracy = tempAccuracy;
  end
end




