function [ Ypred ] = evalCluster(Cluster, Xtest)
  [~, testClusterAssignments] = min(distance(Cluster.centroids, Xtest));
  Ypred = Cluster.centroidLabels(testClusterAssignments);
end