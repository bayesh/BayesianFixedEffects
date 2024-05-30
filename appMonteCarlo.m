% Replication code for the paper 
% "Bayesian Estimation of Fixed Effects Models with Large Datasets"
% Section 4: Monte Carlo Study
% Comparison between the Gibbs sampler and alternative methods for 
% estimating multi-way fixed effects models, including
% o Least Squares Dummy variable (LSDV)
% o Alternating projections of Gaure (2013)
% o Zigzag Gauss-Seidel iterative algorithm of Guimaraes and Portugal (2010)
% o Gibbs sampler of Chib and Carlin (1999) for random effects
%
% References:
%
% Chib, S. and B. P. Carlin (1999). On MCMC sampling in hierarchical 
% longitudinal models. Statistics and Computing 9, 17-26. 2, 4.
%
% Gaure, S., 2013. OLS with multiple high dimensional category variables. 
% Computational Statistics and Data Analysis, 66, 8-18.
%
% Guimaraes, P. , Portugal, P. , 2010. A simple feasible alternative 
% procedure to estimate models with high-dimensional fixed effects. 
% Stata Journal, 10 (4), 628-649.

clear
clc
rng('Default')

% Model specification
T = 10;
N = 100;
J = 10;

% Data generating process for covariates and fixed effects with correlations 
alpha = randn(1,N,1);
gamma = randn(1,1,J);
delta = randn(T,1,1);
X1 = alpha + randn(T,N,J);
X2 = gamma + randn(T,N,J);
X3 = delta + randn(T,N,J);
X = [X1(:),X2(:),X3(:)];

% Simulate response variables by Equation (1) of the paper
Y = X1 + X2 + X3 + alpha + gamma + delta + 3*randn(T,N,J);
y = Y(:);

% Group indices for fixed effects
Group1 = repmat((1:T)',[1,N,J]);
Group2 = repmat(1:N,[T,1,J]);
Group3 = repmat(1:J,[T*N,1]);
Groups = [Group1(:),Group2(:),Group3(:)];

% Our methods can handle unbalanced (N,T,J), in which some data points are
% missing at random, and missing values are marked as NaNs
omit = 1:7:T*N*J;
y(omit) = NaN;
X(omit,:) = NaN;

% Ground-truth Least Squares Dummy variable (LSDV) estimation
% Code dependency: fitlm functionality in Statistics and Machine Learning Toolbox 
Dummy1 = Groups(:,1) == unique(Groups(:,1))';
Dummy2 = Groups(:,2) == unique(Groups(:,2))';
Dummy3 = Groups(:,3) == unique(Groups(:,3))';
varNames = ["x1","x2","x3","DummyT"+(1:T-1),"DummyN"+(1:N-1),"DummyJ"+(1:J-1),"y"];
MdlThreeWay = fitlm([X,Dummy1(:,1:end-1),Dummy2(:,1:end-1),Dummy3(:,1:end-1)],y,'VarNames',varNames);
estimatorLSDV = MdlThreeWay.Coefficients.Estimate(1:4);
seLSDV = MdlThreeWay.Coefficients.SE(1:4);

% Gibbs sampler for fixed effects
burnIn = 500;
BetaDrawsFE = GibbsFixedEffects(X,y,Groups,'BurnIn',0);
estimatorFE = mean(BetaDrawsFE(:,burnIn+1:end),2);
seFE = std(BetaDrawsFE(:,burnIn+1:end),0,2);

% Zigzag Gauss-Seidel iterative algorithm of Guimaraes and Portugal (2010)
% implemented by the Gibbs sampler under an infinitesimal sigma2
BetaDrawsS0 = GibbsFixedEffects(X,y,Groups,'Sigma2',eps,'BurnIn',0);
estimatorS0 = BetaDrawsS0(:,end);

% Gibbs sampler of Chib and Carlin (1999) for random effects
% Due to correlations and endogeneity, random effects estimators are not consistent
BetaDrawsRE = GibbsRandomEffects(X,y,Groups,'BurnIn',0);
estimatorRE = mean(BetaDrawsRE(:,burnIn+1:end),2);
seRE = std(BetaDrawsRE(:,burnIn+1:end),0,2);

% Alternating projections of Gaure (2013)
[estimatorAP,~,CovAP,BetaDrawsAP] = AlternatingProjections(X,y,Groups);
seAP = sqrt(diag(CovAP));

% Summary of estimators of five methods
% See Table 1 of the paper
AIO = zeros(8,5);
AIO(1:2:end,:) = [estimatorLSDV,estimatorAP,estimatorS0,estimatorFE,estimatorRE];
AIO(2:2:end,:) = [seLSDV,seAP,NaN(4,1),seFE,seRE];
AIO = AIO(3:end,:); % suppress intercept display
colNames = ["LSDV","AP","Zigzag","Gibbs","Random"];
rowNames = ["Beta1";" ";"Beta2";" ";"Beta3";" "];
try
    internal.econ.tableprint(AIO,'FileName','Table1.xlsx','ColNames',colNames,'RowNames',rowNames,'bracketData',2:2:8);
catch
    disp(array2table(AIO,'VariableNames',colNames))    
end

% Plot intermediate Beta values at each iteration
% See Figure 1 of the paper
figure(1)
iterations = 1:100;
varInd = 4;
plot(iterations,repmat(estimatorLSDV(varInd),1,numel(iterations)),'-b',...
    iterations,BetaDrawsAP(varInd,iterations),'-.b',...
    iterations,BetaDrawsS0(varInd,iterations),'--r',...
    iterations,BetaDrawsFE(varInd,iterations),'.c')
legend('LSDV','AP','Zigzag','Gibbs')
xlabel('Iterations')
ylabel('\beta_{3}')
ylim([0.8 1.3])


%-------------------------------------------------------------------------
function [Beta,sigma2,BetaCov,BetaSeq,Summary] = AlternatingProjections(X,y,Groups,varargin)

% Parse optional name-value arguments
callerName = 'AlternatingProjections';
parseObj = inputParser;
addParameter(parseObj,'Intercept',true,@(x)validateattributes(x,{'numeric','logical'},{'scalar','binary'},callerName));
addParameter(parseObj,'MaxIterations',1000,@(x)validateattributes(x,{'numeric'},{'scalar','positive','integer'},callerName));
addParameter(parseObj,'Tolerance',1e-3,@(x)validateattributes(x,{'numeric'},{'scalar','positive'},callerName));
addParameter(parseObj,'VarNames','',@(x)validateattributes(x,{'cell','string'},{},callerName));
addParameter(parseObj,'Display',true,@(x)validateattributes(x,{'numeric','logical'},{'scalar','binary'},callerName));
parse(parseObj,varargin{:});
intercept = parseObj.Results.Intercept;
maxIter = parseObj.Results.MaxIterations;
tol = parseObj.Results.Tolerance;
varNames = parseObj.Results.VarNames;
dispFlag = parseObj.Results.Display;

% Check input sizes
numObs = size(y,1);
[numObsX,numX] = size(X);
[numObsG,numGroups] = size(Groups);
if numObs~=numObsX || numObs~=numObsG
    error('X, y and Groups must have the same number of rows (observations).')
end

% Remove missing values
bad = isnan(y) | any(isnan(X),2) | any(ismissing(Groups),2);
y = y(~bad);
X = X(~bad,:);
Groups = Groups(~bad,:);
numObs = size(y,1);

% Generate variable names
if isempty(varNames)
    varNames = "x" + (1:numX);
else
    varNames = string(varNames);
    varNames = varNames(:)';
end
if intercept
    varNames = ["Intercept",varNames];
end

% Add an intercept term to X
if intercept
    X = [ones(numObs,1),X]; 
    numX = numX + 1;
end

% Re-code group indices as integers 1, 2, 3,...
Integers = zeros(numObs,numGroups);
for g = 1:numGroups
    Integers(:,g) = findgroups(Groups(:,g));
end
numLevels = max(Integers);
numDummy = sum(numLevels) - numGroups;

% Cache selected indices
Selection = cell(max(numLevels),numGroups);
for g = 1:numGroups
    group = Integers(:,g);    
    for n = 1:numLevels(g)
        Selection{n,g} = find(group==n);
    end
end

% Alternating projections for each group sequentially, iterating until convergence
YX = [y,X];
BetaSeq = zeros(numX,maxIter);
for r = 1:maxIter
    Backup = YX;
    for g = 1:numGroups
        for n = 1:numLevels(g)-1       % drop one level for normalization
            select = Selection{n,g};   % load cached selection indices
            DataSelect = YX(select,:);
            YX(select,:) = DataSelect - mean(DataSelect); % Within-transform 
        end
    end

    % Intermediate estimation results on Beta at each iteration
    Py = YX(:,1);
    PX = YX(:,2:end);
    BetaSeq(:,r) = PX \ Py;

    % Check convergence
    if norm(YX-Backup,1) < tol
        fprintf('Alternating projections converged after %d iterations.\n',r);
        BetaSeq(:,r+1:end) = repmat(BetaSeq(:,r),1,maxIter-r);
        break
    end
end

% Alternating projections estimator by OLS of transformed data
Py = YX(:,1);
PX = YX(:,2:end);
Beta = PX \ Py;

% Residuals and estimator covariance matrix (with degree of freedom adjustment)
resid = Py - PX * Beta;
RSS = resid' * resid;
sigma2 = RSS / (numObs-numX-numDummy);
invXX = inv(PX'*PX);
BetaCov = sigma2 .* invXX;
BetaSE = sqrt(diag(BetaCov));
tStat = Beta ./ BetaSE;
pValue = erfc(0.7071*abs(tStat));

% Display estimation results
Summary = array2table([Beta,BetaSE,tStat,pValue],'Variablenames',["estimator","se","tStat","pValue"],'Rownames',varNames);
if dispFlag
    disp(Summary)
end
end


function [BetaDraws,sigma2Draws,FixedEffectMean,Alpha,Summary] = GibbsFixedEffects(X,y,Groups,varargin)

% Parse optional name-value arguments
callerName = 'GibbsFixedEffects';
parseObj = inputParser;
addParameter(parseObj,'Intercept',true,@(x)validateattributes(x,{'numeric','logical'},{'scalar','binary'},callerName));
addParameter(parseObj,'NumDraws',10000,@(x)validateattributes(x,{'numeric'},{'scalar','positive','integer'},callerName));
addParameter(parseObj,'BurnIn',1000,@(x)validateattributes(x,{'numeric'},{'scalar','nonnegative','integer'},callerName));
addParameter(parseObj,'Sigma2',NaN,@(x)validateattributes(x,{'numeric'},{'scalar','nonnegative'},callerName));
addParameter(parseObj,'VarNames','',@(x)validateattributes(x,{'cell','string'},{},callerName));
addParameter(parseObj,'Display',true,@(x)validateattributes(x,{'numeric','logical'},{'scalar','binary'},callerName));
parse(parseObj,varargin{:});
intercept = parseObj.Results.Intercept;
numDraws = parseObj.Results.NumDraws;
burnIn = parseObj.Results.BurnIn;
sigma2Fixed = parseObj.Results.Sigma2;
varNames = parseObj.Results.VarNames;
dispFlag = parseObj.Results.Display;

% Check input sizes
numObs = size(y,1);
[numObsX,numX] = size(X);
[numObsG,numGroups] = size(Groups);
if numObs~=numObsX || numObs~=numObsG
    error('X, y and Groups must have the same number of rows (observations).')
end

% Remove missing values
bad = isnan(y) | any(isnan(X),2) | any(ismissing(Groups),2);
y = y(~bad);
X = X(~bad,:);
Groups = Groups(~bad,:);
numObs = size(y,1);

% Generate variable names
if isempty(varNames)
    varNames = "x" + (1:numX);
else
    varNames = string(varNames);
    varNames = varNames(:)';
end
if intercept
    varNames = ["Intercept",varNames];
end

% Add an intercept term to X
if intercept
    X = [ones(numObs,1),X]; 
    numX = numX + 1;
end

% Re-code group indices as integers 1, 2, 3,...
Integers = zeros(numObs,numGroups);
for g = 1:numGroups
    Integers(:,g) = findgroups(Groups(:,g));
end
numLevels = max(Integers);
% numDummy = sum(numLevels) - numGroups;

% Cache selected indices
Selection = cell(max(numLevels),numGroups);
for g = 1:numGroups
    group = Integers(:,g);    
    for n = 1:numLevels(g)
        Selection{n,g} = find(group==n);
    end
end

% Initialize fixed effects
FixedEffects = zeros(numObs,numGroups,'like',y);
sumFixedEffect = sum(FixedEffects,2);

% Sufficient statistics
XX = X'*X;

% Initialize Beta
Beta = XX \ (X'*y);

% Prepare for Gibbs sampling
BetaDraws = zeros(numX,numDraws,'like',y);
sigma2Draws = zeros(1,numDraws,'like',y);
FixedEffectMean = FixedEffects;
numDrawsAdjust = numDraws + burnIn;
count = 0;
Hstatus = waitbar(0,'Gibbs Sampling');
cleanUp = onCleanup(@()close(Hstatus));
percentage = floor(numDrawsAdjust / 100);
for r = 1:numDrawsAdjust

    if mod(r,percentage) == 0
        waitbar(r/numDrawsAdjust,Hstatus,'Gibbs Sampling');
    end    
    
    % Conditional on fixed effects, update Beta and sigma2
    % by conjugate Bayesian linear regressions
    % y-D*Alpha = X*Beta + sigma*error
    yDA = y - sumFixedEffect;
    [Beta,sigma2] = simulateRegressionDiffuse(yDA,X,XX,Beta,sigma2Fixed);

    % Conditional on Beta and sigma2, update fixed effects sequentially,
    % given the remaining elements in FixedEffects matrix
    yXB = y - X*Beta;
    for g = 1:numGroups        
        % Response variable for updating each fixed effect is the residual
        fixedEffect = FixedEffects(:,g);
        otherEffect = sumFixedEffect - fixedEffect;
        resid = yXB - otherEffect;
        
        % Posterior conditional distributions are normal        
        for n = 1:numLevels(g)-1       % drop one level for normalization            
            select = Selection{n,g};   % load cached selection indices
            fixedEffect(select) = mean(resid(select)) + sqrt(sigma2/numel(select)) * randn;                  
        end
        FixedEffects(:,g) = fixedEffect;
        sumFixedEffect = fixedEffect + otherEffect;        
    end

    % Record draws
    if r > burnIn
        count = count + 1;
        BetaDraws(:,count) = Beta;
        sigma2Draws(:,count) = sigma2;
        FixedEffectMean = (1-1/count)*FixedEffectMean + 1/count*FixedEffects;
    end
end

% Re-format the matrix FixedEffectMean as a cell array of unique elements
Alpha = cell(1,numGroups);
for g = 1:numGroups
    alpha = zeros(numLevels(g),1);
    for n = 1:numLevels(g)
        select = find(Integers(:,g)==n,1);
        alpha(n) = FixedEffectMean(select,g);
    end
    Alpha{g} = alpha;
end

% Summary statistics
BetaMean = mean(BetaDraws,2);
BetaStd = std(BetaDraws,0,2);
sigma2Mean = mean(sigma2Draws,2);
sigma2Std = std(sigma2Draws,0,2);
Summary = array2table([[BetaMean;sigma2Mean],[BetaStd;sigma2Std]],'Variablenames',["mean","std"],'Rownames',[varNames,"sigma2"]);
if dispFlag
    disp(Summary)
end
end


% Posterior simulation of conjugate Bayesian linear regression
function [Beta,sigma2] = simulateRegressionDiffuse(y,X,XX,Beta,sigma2Fixed)

% Sufficient statistics
[numObs,numX] = size(X);
XY = X'*y;
YY = y'*y;

% Simulate sigma2 from posterior conditional distribution
if isnan(sigma2Fixed)
    RSS = YY - 2*Beta'*XY + Beta'*XX*Beta;
    A = numObs/2;
    B = RSS/2;
    sigma2 = 1 ./ gamrnd(A, 1/B);
else
    sigma2 = sigma2Fixed;
end

% Simulate Beta from posterior conditional distribution
Mu = XX \ XY;
Precision = XX;
PrecisionChol = chol(Precision);
Beta = Mu + sqrt(sigma2) .* (PrecisionChol \ randn(numX,1));
end


function [BetaDraws,sigma2Draws,sigma2EffectsDraws,RandomEffectMean,Alpha,Summary] = GibbsRandomEffects(X,y,Groups,varargin)

% Parse optional name-value arguments
callerName = 'GibbsRandomEffects';
parseObj = inputParser;
addParameter(parseObj,'Intercept',true,@(x)validateattributes(x,{'numeric','logical'},{'scalar','binary'},callerName));
addParameter(parseObj,'NumDraws',10000,@(x)validateattributes(x,{'numeric'},{'scalar','positive','integer'},callerName));
addParameter(parseObj,'BurnIn',1000,@(x)validateattributes(x,{'numeric'},{'scalar','nonnegative','integer'},callerName));
addParameter(parseObj,'PriorA',3,@(x)validateattributes(x,{'numeric'},{'positive'},callerName));
addParameter(parseObj,'PriorB',1,@(x)validateattributes(x,{'numeric'},{'positive'},callerName));
addParameter(parseObj,'Sigma2',NaN,@(x)validateattributes(x,{'numeric'},{'scalar','nonnegative'},callerName));
addParameter(parseObj,'VarNames','',@(x)validateattributes(x,{'cell','string'},{},callerName));
addParameter(parseObj,'Display',true,@(x)validateattributes(x,{'numeric','logical'},{'scalar','binary'},callerName));
parse(parseObj,varargin{:});
intercept = parseObj.Results.Intercept;
numDraws = parseObj.Results.NumDraws;
burnIn = parseObj.Results.BurnIn;
priorA = parseObj.Results.PriorA;
priorB = parseObj.Results.PriorB;
sigma2Fixed = parseObj.Results.Sigma2;
varNames = parseObj.Results.VarNames;
dispFlag = parseObj.Results.Display;

% Check input sizes
numObs = size(y,1);
[numObsX,numX] = size(X);
[numObsG,numGroups] = size(Groups);
if numObs~=numObsX || numObs~=numObsG
    error('X, y and Groups must have the same number of rows (observations).')
end

% Remove missing values
bad = isnan(y) | any(isnan(X),2) | any(ismissing(Groups),2);
y = y(~bad);
X = X(~bad,:);
Groups = Groups(~bad,:);
numObs = size(y,1);

% Generate variable names
if isempty(varNames)
    varNames = "x" + (1:numX);
else
    varNames = string(varNames);
    varNames = varNames(:)';
end
if intercept
    varNames = ["Intercept",varNames];
end

% Add an intercept term to X
if intercept
    X = [ones(numObs,1),X]; 
    numX = numX + 1;
end

% Re-code group indices as integers 1, 2, 3,...
Integers = zeros(numObs,numGroups);
for g = 1:numGroups
    Integers(:,g) = findgroups(Groups(:,g));
end
numLevels = max(Integers);
% numDummy = sum(numLevels) - numGroups;

% Cache selected indices
Selection = cell(max(numLevels),numGroups);
for g = 1:numGroups
    group = Integers(:,g);    
    for n = 1:numLevels(g)
        Selection{n,g} = find(group==n);
    end
end

% Initialize random effects
RandomEffects = zeros(numObs,numGroups,'like',y);
sumRandomEffect = sum(RandomEffects,2);
sigma2Effects = ones(numGroups,1);
AlphaMat = zeros(max(numLevels),numGroups);

% Hyperparameters A and B of inverse gamma prior IG(A,B)
if isscalar(priorA)
    priorA = repmat(priorA,1,numGroups+1);
end

if isscalar(priorB)
    priorB = repmat(priorB,1,numGroups+1);
end

% Sufficient statistics
XX = X'*X;

% Initialize Beta
Beta = XX \ (X'*y);

% Prepare for Gibbs sampling
BetaDraws = zeros(numX,numDraws,'like',y);
sigma2Draws = zeros(1,numDraws,'like',y);
sigma2EffectsDraws = zeros(numGroups,numDraws,'like',y);
RandomEffectMean = RandomEffects;
numDrawsAdjust = numDraws + burnIn;
count = 0;
Hstatus = waitbar(0,'Gibbs Sampling');
cleanUp = onCleanup(@()close(Hstatus));
percentage = floor(numDrawsAdjust / 100);
for r = 1:numDrawsAdjust

    if mod(r,percentage) == 0
        waitbar(r/numDrawsAdjust,Hstatus,'Gibbs Sampling');
    end    
    
    % Conditional on random effects, update Beta and sigma2
    % by conjugate Bayesian linear regressions
    % y-D*Alpha = X*Beta + sigma*error
    yDA = y - sumRandomEffect;
    [Beta,sigma2] = simulateRegression(yDA,X,XX,Beta,sigma2Fixed,priorA(end),priorB(end));

    % Conditional on Beta and sigma2, update random effects sequentially,
    % given the remaining elements in RandomEffects matrix
    yXB = y - X*Beta;
    for g = 1:numGroups        
        % Response variable for updating each random effect is the residual
        randomEffect = RandomEffects(:,g);
        otherEffect = sumRandomEffect - randomEffect;
        resid = yXB - otherEffect;
        
        % Posterior conditional distributions are normal        
        for n = 1:numLevels(g)           
            select = Selection{n,g};   % load cached selection indices
            precision = 1/sigma2Effects(g) + numel(select)/sigma2;
            condVar = 1/precision;
            condMean = sum(resid(select))/sigma2 / precision;
            AlphaMat(n,g) = condMean + sqrt(condVar) * randn;
            randomEffect(select) = AlphaMat(n,g);                 
        end
        RandomEffects(:,g) = randomEffect;
        sumRandomEffect = randomEffect + otherEffect;

        % Variances of random effects follow inverse gamma distributions
        IGa = priorA(g) + numLevels(g)/2;
        IGb = priorB(g) + AlphaMat(:,g)'*AlphaMat(:,g)/2;
        sigma2Effects(g) = 1/gamrnd(IGa, 1/IGb);
    end

    % Record draws
    if r > burnIn
        count = count + 1;
        BetaDraws(:,count) = Beta;
        sigma2Draws(:,count) = sigma2;
        sigma2EffectsDraws(:,count) = sigma2Effects;
        RandomEffectMean = (1-1/count)*RandomEffectMean + 1/count*RandomEffects;
    end
end

% Re-format the matrix RandomEffectMean as a cell array of unique elements
Alpha = cell(1,numGroups);
for g = 1:numGroups
    alpha = zeros(numLevels(g),1);
    for n = 1:numLevels(g)
        select = find(Integers(:,g)==n,1);
        alpha(n) = RandomEffectMean(select,g);
    end
    Alpha{g} = alpha;
end

% Summary statistics
BetaMean = mean(BetaDraws,2);
BetaStd = std(BetaDraws,0,2);
sigma2Mean = mean(sigma2Draws,2);
sigma2Std = std(sigma2Draws,0,2);
sigma2EffectsMean = mean(sigma2EffectsDraws,2);
sigma2EffectsStd = std(sigma2EffectsDraws,0,2);
rowNames = [varNames,"sigma2","sigma2Effects"+(1:numGroups)];
Summary = array2table([[BetaMean;sigma2Mean;sigma2EffectsMean],[BetaStd;sigma2Std;sigma2EffectsStd]],'Variablenames',["mean","std"],'Rownames',rowNames);
if dispFlag
    disp(Summary)
end
end


% Posterior simulation of conjugate Bayesian linear regression
function [Beta,sigma2] = simulateRegression(y,X,XX,Beta,sigma2Fixed,IGa,IGb)

% Sufficient statistics
[numObs,numX] = size(X);
XY = X'*y;
YY = y'*y;

% Simulate sigma2 from posterior conditional distribution
if isnan(sigma2Fixed)
    RSS = YY - 2*Beta'*XY + Beta'*XX*Beta;
    A = IGa + numObs/2;
    B = IGb + RSS/2;
    sigma2 = 1 ./ gamrnd(A, 1/B);
else
    sigma2 = sigma2Fixed;
end

% Simulate Beta from posterior conditional distribution
Mu = XX \ XY;
Precision = XX;
PrecisionChol = chol(Precision);
Beta = Mu + sqrt(sigma2) .* (PrecisionChol \ randn(numX,1));
end
