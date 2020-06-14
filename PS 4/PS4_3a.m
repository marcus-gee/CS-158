% 3) Logistic Regression Analysis of the Stock Market Data
% a.

% data
sm_train = table2array(readtable("data/stock_market_train.csv","HeaderLines",1));
sm_test  = table2array(readtable("data/stock_market_test.csv", "HeaderLines",1));
ps4problem3a(sm_train, sm_test);

function ps4problem3a(train, test) 
    [N,p_tr] = size(train);
    [n,p_te] = size(test);
    
    % bias column
    train = [ones([N,1]), train];
    test  = [ones([n,1]), test];
    
    % X"s and y"s
    X_train = train(:,1:end-1);
    X_test  = test(:,1:end-1);

    % G_0 is Direction= +1 and G_1 is Direction= -1
    % so represent y_i"s for G_0 = 1 and G_1 = 0 (see p.62 of notes)
    y_train = (train(:,end) == 1);
    y_test  = (test(:,end) == 1);
    
    eps = 10^-6;
    Bk  = zeros([p_tr,1]); % current:  B_k
    Bk_1 = ones([p_te,1]); % prev: B_k-1
    i = 0;
    while (norm(Bk - Bk_1)/norm(Bk_1)) > eps
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
    Err = classification_err(y_test, pred) / n % = 0.4960
    
    sig2 = diag(inv(X_train' * W_k * X_train));
    zj   = Bk ./ (sqrt(sig2));
    p_val= 2 * normcdf((-1 .* abs(zj)));
    
    res = [Bk zj p_val];
    rowNames = {'1','Lag1','Lag2','Lag3','Lag4','Lag5','Volume'};
    colNames = {'B','Zscore','pValue'};
    Table = array2table(res,"RowNames",rowNames,"VariableNames",colNames)
       
    %            ____B____   _Zscore_    _pValue_
    % 
    % 1           -0.13528    -0.49814    0.61838
    % Lag1       -0.073533     -1.2893    0.19729
    % Lag2        -0.07272     -1.3169    0.18786
    % Lag3      -0.0086285    -0.15322    0.87823
    % Lag4        0.016658     0.29284    0.76964
    % Lag5        0.057834       1.077    0.28148
    % Volume       0.13287     0.74372    0.45705
end

function err = classification_err(act, pred)
    err = sum(act ~= pred);
end

function prob = p(X,B)
    prob = exp(X * B) ./ (1 + exp(X * B));
end