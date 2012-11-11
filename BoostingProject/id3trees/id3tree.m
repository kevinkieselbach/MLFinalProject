function T=id3tree(xTr,yTr,maxdepth,weights)
% function T=id3tree(xTr,yTr,maxdepth,weights)
%
% The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split). 
% Each example can be weighted with "weights". 
%
% Builds an id3 tree
%
% Input:
% xTr = dxn input matrix with n column-vectors of dimensionality d
% xTe = dxm input matrix with n column-vectors of dimensionality d
% maxdepth = maximum tree depth
% weights = 1xn vector where weights(i) is the weight of example i
% 
% Output:
% T = decision tree 
%


[mapback,~,yTr]=unique(yTr); % make labels 1,2,3,...,C
n=length(yTr);
 if nargin<4,weights=ones(1,n)./n;end;
 if nargin<3,maxdepth=inf;end;
 
 T=zeros(6,3*n);
 Stack{1}.index=1:n;
 Stack{1}.depth=1;
 Stack{1}.id=1;
 Stack{1}.parent=0;
 writer=1;
 
 
 while (~isempty(Stack))
     sl=length(Stack);
     s=Stack{sl};
     parent=s.parent;
     index=s.index;
     x=xTr(:,index);
     y=yTr(index);
     w=weights(index);
     depth=s.depth;
     id=s.id;     
     n=size(x,2);     
     T(6,id)=parent;
     T(1,id)=mapback(wmode(y,w));
     
     if depth==maxdepth || all(y==y(1)) || max(max(abs(diff(x'))))<eps*100,         
         Stack(sl)=[];
     else
         [feature,cut]=entropysplit(x,y,w);
         T(2,id)=feature;
         T(3,id)=cut;
         T(4,id)=writer+1;
         T(5,id)=writer+2;
         
         
         % Left child
         Stack{sl}.index=index(x(feature,:)<=cut); 
         Stack{sl}.id=writer+1;
         Stack{sl}.depth=depth+1;
         Stack{sl}.parent=id;
         
         % Right child
         Stack{sl+1}.index=index(x(feature,:)>cut);
         Stack{sl+1}.id=writer+2;
         Stack{sl+1}.depth=depth+1;
         Stack{sl+1}.parent=id;
         
         % increase the writer
         writer=writer+2;         
     end;
     
 end;
 T=T(:,1:writer);


function m=wmode(y,w)

S=sparse(1:length(y),y,w);
s=sum(S,1);
[v,m]=max(s);


 



