% 5) Classification Trees for Stock Market Data

train = table2array(readtable('data/stock_market_train.csv',"HeaderLines",1));
test  = table2array(readtable('data/stock_market_test.csv', "HeaderLines",1));
ps5problem5(train, test);

function ps5problem5(train, test)
    % a.
    N = 1000;
    n = 250; 
    p = 6;
    
    X_train = train(:, 1:p);
    y_train = train(:, end);
    X_test  = test(:, 1:p);
    y_test  = test(:, end);
    
    % fit tree
    T0 = fitctree(X_train, y_train);
    view(T0,'Mode','graph');
    
    % predictions and error
    T0_train_pred = predict(T0,X_train);
    T0_test_pred  = predict(T0,X_test);
    
    T0_train_err = misclass_err(y_train, T0_train_pred) % = 0.1180
    T0_test_err  = misclass_err(y_test, T0_test_pred)   % = 0.4760
    
    % b.
    % LOOCV to find T
    alpha = 0:0.001:0.04;
    ERR = zeros([N,length(alpha)]);
    for i = 1:N
        % leave one out
        train_ind = [1:i-1, i+1:N];
        xi = X_train(train_ind, :);
        yi = y_train(train_ind, :);
        x_valid = X_train(i, :);
        y_valid = y_train(i, :);
            
        T0 = fitctree(xi, yi);
        % CCP
        for a = alpha
            T = prune(T0,'Alpha',a);
            pred = predict(T,x_valid);
            % Support Vector Machineerrors
            test_err = misclass_err(y_valid,pred);
            ERR(i, round(a/0.001)+1) = test_err; 
        end
    end
    CV_mean = mean(ERR, 1);
        
    [~, min_alpha] = min(CV_mean);
    min_alpha = (min_alpha-1) / 1000 % = 0.0100
    
    % c.
    T = prune(T0,'Alpha',min_alpha);
    view(T,'Mode','graph');
    
    T_train_pred = predict(T,X_train);
    T_test_pred  = predict(T,X_test);
    
    T_train_err = misclass_err(y_train, T_train_pred) % = 0.4010
    T_test_err  = misclass_err(y_test, T_test_pred)   % = 0.5840
end

function err = misclass_err(act, pred)
    err = sum(act ~= pred) / length(pred);
end