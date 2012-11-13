function [ Ypred ] = evallogiboost(B, Xtest)
    Ypred = zeros(1,size(Xtest,2));
    for i = 1:size(B,2)
        Ypred = Ypred + B{i}.weight * evaltree(B{i}.tree,Xtest);
    end
    Ypred(Ypred>=0.5) = 1;
    Ypred(Ypred<0.5) = 0;
    
end