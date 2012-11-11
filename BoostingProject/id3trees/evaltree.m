function [ypredict]=evaltree(T,xTe)
% function [ypredict]=evaltree(T,xTe);
%
% input: 
% T0 : tree structure
% xTe : Test data (dxn matrix)
%
% output: 
%
% ypredict : predictions of labels for xTe
%

%% fill in code here
%%<<kqw

n=size(xTe,2);
ypredict=zeros(1,n);
  for i=1:n
    index=1;
	while (true)        
		if T(4,index)==0,ypredict(i)=T(1,index);break;end;
	 	if xTe(T(2,index),i)<=T(3,index),           
            index=T(4,index);
        else index=T(5,index);end;        
	end;
  end;

%%>>kqw