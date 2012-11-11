function [ Ypred ] = evalboost(B, Xtest)
    Ypred = zeros(1,size(Xtest,2));
    for i = 1:size(B,2)
        Ypred = Ypred + B{i}.weight * evaltree(B{i}.tree,Xtest);
    end
end