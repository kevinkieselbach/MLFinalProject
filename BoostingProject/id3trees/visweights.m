function visdata(xTr,yTr,weights)
% function visweights
% 
% visualizes a weighted data set and a tree of depth 4
%

if nargin<3,weights=ones(1,length(yTr))./length(yTr);end;
s=scatter(xTr(1,:),xTr(2,:),weights.*10000,yTr,'Marker','.');
T=id3tree(xTr,yTr,4,weights);
vistree(T);
