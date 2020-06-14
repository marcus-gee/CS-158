% 4) Regression Trees for Boston Housing Data

train = table2array(readtable('data/Boston_train.csv',"HeaderLines",0));
test  = table2array(readtable('data/Boston_test.csv', "HeaderLines",0));
ps5problem4(train, test);

function ps5problem4(train, test)
    % a.
    N = 400;
    n = 106; 
    p = 13;
    
    X_train = train(:, 1:p);
    y_train = train(:, end);
    X_test  = test(:, 1:p);
    y_test  = test(:, end);
    
    % fit tree
    T0 = fitrtree(X_train, y_train);
    view(T0,'Mode','graph');
    
    % predictions and error
    T0_train_pred = predict(T0,X_train);
    T0_test_pred  = predict(T0,X_test);
    
    T0_train_err = RSS(y_train, T0_train_pred) % = 2.1707
    T0_test_err  = RSS(y_test, T0_test_pred)   % = 12.9867
    
    % b.
    % LOOCV to find T
    alpha = 0:0.1:2;
    ERR = zeros([N,length(alpha)]);
    for i = 1:N
        % leave one out
        train_ind = [1:i-1, i+1:N];
        xi = X_train(train_ind, :);
        yi = y_train(train_ind, :);
        x_valid = X_train(i, :);
        y_valid = y_train(i, :);
            
        T0 = fitrtree(xi, yi);
        % CCP
        for a = alpha
            T = prune(T0,'Alpha',a);
            pred = predict(T,x_valid);
            % errors
            test_err = RSS(y_valid,pred);
            ERR(i, round(a/0.1)+1) = test_err; 
        end
    end
    CV_mean = mean(ERR, 1);
        
    [~, min_alpha] = min(CV_mean);
    min_alpha = (min_alpha-1) / 10 % = 0.70
    
    % c.
    T = prune(T0,'Alpha',min_alpha);
    view(T,'Mode','graph');
    
    T_train_pred = predict(T,X_train);
    T_test_pred  = predict(T,X_test);
    
    T_train_err = RSS(y_train, T_train_pred) % = 10.2121
    T_test_err  = RSS(y_test, T_test_pred)   % = 9.6135
end

function rss = RSS(act, pred)
    rss = sum( (act - pred).^2 ) / length(pred);
end