data = load('coil.mat');
X = data.xTr;
Y = data.yTr;
Xt = data.xTe;
Yt = data.yTe;

[~,~,fun] = svm_kernel('X', X', 'Y', Y', 'sigma', 10);
labels = sign(fun(Xt'));
accuracy = sum(labels - Yt' == 0)/size(labels,1);
display(accuracy)