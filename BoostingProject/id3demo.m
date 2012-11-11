function [x, y,W] = boostid3demo(varargin)
% function [trainPoints, trainLabels,W] = boost_linear(varargin)
%

pars.X=[];
pars.Y=[];
pars.C=0.01;
pars.maxdepth=100;
pars.viscolor=true;
pars.vismargin=false;
pars.maxiter=10;
pars=extractpars(varargin,pars);


% Define the symbols and colors we'll use in the plots later
symbols = {'o','x'};
classvals = [-1 1];
% Initialize training data to empty; will get points from user
% Obtain points froom the user:
trainPoints=pars.X;
trainLabels=pars.Y;
clf;
axis([-5 5 -5 5]);
if isempty(trainPoints)
	trainLabels=[];


    hold on; % Allow for overwriting existing plots
    xlim([-5 5]); ylim([-5 5]);
    
    for c = 1:2
        title(sprintf('Click to create points from class %d. Press enter when finished.', c));
        [x y] = getpts;
        
        plot(x,y,symbols{c},'LineWidth', 2, 'Color', 'black');
        
        % Grow the data and label matrices
        trainPoints = vertcat(trainPoints, [x y]);
        trainLabels = vertcat(trainLabels, repmat(classvals(c), numel(x), 1));        
    end

end

% To evaluate the classifier, we discretize the range of inputted values
% into a grid of points that we will test on to produce the decision
% boundaries

% meshgrid returns a matrix of X and Y values for every point in each
% combination of the X range and Y range that we specify; we string these
% 80x80 matrices out into a big 6400x2 matrix that we'll feed into the K-NN
% function:


% shuffle training points randomly
i=randperm(length(trainLabels));
trainPoints=trainPoints(i,:);
trainLabels=trainLabels(i);

% if strcmp(method, 'knn')
%     testLabels = sign(knn_test(K, trainPoints,trainLabels,testPoints,distFunc));
% elseif strcmp(method, 'kernreg')
%     testLabels = sign(kernreg_test(K, trainPoints,trainLabels,testPoints,distFunc));
% else
%     error('Unrecognized method');
% end

% initialize with all-zero


pars.bias=1; % alwyas switch on bias;
x=trainPoints;
y=trainLabels;

T=id3tree(x',(y'+1)./2+1,pars.maxdepth);
fun=@(x) (evaltree(T,x')'-1).*2-1;
visdecision(x(:,1:2),y,fun,'viscolor',pars.viscolor,'vismargin',pars.vismargin);

