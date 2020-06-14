% 3) Logistic Regression Analysis of the Stock Market Data
% b.

% data
sm_train = table2array(readtable("data/stock_market_train.csv","HeaderLines",1));
sm_test  = table2array(readtable("data/stock_market_test.csv", "HeaderLines",1));
ps4problem3b(sm_train, sm_test);

function ps4problem3b(train, test) 
    N = length(train);
    n = length(test);
    
    % bias column
    train = [ones([N,1]), train];
    test  = [ones([n,1]), test];
    
    % X's and y's
    % refit with two most significant predictors
    % from part (a), that is Lag1 and Lag 2
    X_train = train(:,1:3);
    X_test  = test(:,1:3);
    
    % G_0 is Direction= +1 and G_1 is Direction= -1
    % so represent y's as 0's and 1's for corresponding G
    y_train = (train(:,end) == -1);
    y_test  = (test(:,end) == -1);
    
    [~,p_tr] = size(X_train);
    [~,p_te] = size(X_test);
    
    eps = 10^-5;
    Bk  = zeros([p_tr,1]); % current:  B_k
    Bk_1 = ones([p_te,1]); % prev: B_k-1
    i = 0;
    while norm(Bk - Bk_1) > eps
        % cond. prob matrix
        p_k = p(X_train, Bk);
        % weight matrix
        W_k = (p_k .* (1 - p_k)) .* eye(N);
        % update betas
        Bk_1 = Bk;
        Bk = Bk + ((X_train' * W_k * X_train) \ X_train' * (y_train - p_k));
        i = i+1;
    end
    % one more update after stopping
    p_k = p(X_train, Bk);
    W_k = (p_k .* (1 - p_k)) .* eye(N);
    
    % make predictions
    pred = round(p(X_test,Bk));
    % find error
    Err = classification_err(y_test, pred) / n % = 0.4720
    
    % classify correctness/errors of predictions
    n00 = sum((pred == 0) & (y_test == 0));
    n01 = sum((pred == 0) & (y_test == 1));
    n11 = sum((pred == 1) & (y_test == 1));
    n10 = sum((pred == 1) & (y_test == 0));
    
    Err_0 = n01 / (n00 + n01) % = 0.4603
    Err_1 = n10 / (n11 + n10) % = 0.5082
    
    % if the model predicts the market will be Up (class G_0), i would 
    % recommend buying. this is due to the fact that based on the test 
    % Err_0, when the model predicts the market is Up, it is correct at 
    % least half the time, so long term buying when the market is Up
    % should make you a profit, since you when the market is Up, you'd
    % expect it to stay Up and your stocks will grow in value.
    
    % if the model predicts the market will be Down (class G_1), i would 
    % recommend avoiding any trades. this is due to the fact that based on 
    % the test Err_1, when the model predicts the market is Down, it is 
    % basically a coin flip for it to be correct (50/50). thus, long term,
    % you will likely not make a profit buying or selling, so you 
    % should just stay out. 
end

function err = classification_err(act, pred)
    err = sum(act ~= pred);
end

function prob = p(X,B)
    prob = exp(X * B) ./ (1 + exp(X * B));
end