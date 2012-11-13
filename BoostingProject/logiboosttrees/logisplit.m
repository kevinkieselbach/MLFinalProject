function [feature,cut]=logisplit(xTr,yTr,weights,varargin)
% function [feature,cut]=entropysplit(xTr,yTr,weights);
% 
% Finds the best (feature,cutoff) pair to split data, minimizing square loss.
% 
% INPUT:
% xTr : dxn matrix of n column vectors with d dimensions
% yTr : 1xn row vector of labels
% weights : 1xn vector of where weight(i) is the weight of example i
%
% OUTPUT:
% feature : best feature to split
% cut : Value to split on.
%
%
%
[d,n]=size(xTr);
numSamples = d;

if nargin > 3
    numSamples = varargin{1};
end
d = randsample(d,numSamples,false)';

if exist('weights')~=1,weights=ones(1,n)./n;end;
weights=weights./sum(weights); % Weights need to sum to one
[revmap,b,yTr]=unique(yTr);	
Hbest=Inf;
for i=d
        % 2 classes - > 2 columsn
        % each column is 1 example
        [sx,ii]=sort(xTr(i,:));
        %get ith feature and sort by value of feature
        % ii has indexes from unsorted to sorted (map)
		y=yTr(ii);
        ws=weights(ii);
        
        % get average weighted y values per each possible split from left
        Ybar = cumsum(y .* ws)./cumsum(ws);
        
        Y = cumsum((y.^2).*ws) ./cumsum(ws);
        
        %loss from left 
        Ll = Y - Ybar.^2;
        
        % get average weighted y values per each possible split from right
        YbarR = cumsum(y(end:-1:1) .* ws(end:-1:1))./cumsum(ws(end:-1:1));
        
        YR = cumsum((y(end:-1:1).^2).*ws(end:-1:1))./cumsum(ws(end:-1:1));
        
        %loss from left 
        Lr = YR - YbarR.^2;
        
        %reverse
        Lr = Lr(end:-1:1,:);

%  	  	Y=sparse(1:n,y,ws);
% 
%         S1=spdiags(cumsum(ws)',0,n,n);
%         S1f=spdiags(1./cumsum(ws'),0,n,n);
% 
%         
%         P=S1f*cumsum(Y);
% 		Q=-P.*log2(P);Q(isnan(Q))=0;				
%         Q=S1*Q;
%         
%         S1ud=spdiags(cumsum(ws(end:-1:1)'),0,n,n);
%         S1fud=spdiags(1./cumsum(ws(end:-1:1)'),0,n,n);
% 
%         
%         P2=S1fud*cumsum(Y(end:-1:1,:));
%         
% 		Q2=-P2.*log2(P2);
%         Q2(isnan(Q2))=0;
%         Q2=S1ud*Q2;
% 		Q2=Q2(end:-1:1,:);

        Lr = Lr';
        Ll = Ll';
        
		idif=find(abs(diff(sx))>eps*100); % find places where the features differ
		H=Ll(idif)+Lr(idif+1);
		if isempty(H),continue;end;
		[hb,jc]=min(H);
		if hb<=Hbest, 
		 	cut=mean([sx(idif(jc)) sx(idif(jc)+1)]); % cut <=T.cut		    
			feature=i;
			Hbest=hb;
		end;
end;


