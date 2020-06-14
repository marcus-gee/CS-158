% 1) K-NN and Linear Regression for Regression.

% data
train1 = table2array(readtable('data/dataset1_train.csv', 'HeaderLines',1));
test1  = table2array(readtable('data/dataset1_test.csv', 'HeaderLines',1)); 
train2 = table2array(readtable('data/dataset2_train.csv', 'HeaderLines',1));
test2  = table2array(readtable('data/dataset2_test.csv', 'HeaderLines',1));

% error ratios
R1 = ps1problem1(train1, test1)
% R1 = 8.0073
% bc R2 > 1, the linear regression model outperformed the k_nn model

R2 = ps1problem1(train2, test2)
% R2 = 0.2136
% bc R2 < 1, the k_nn model outperformed the linear regression model


% a) k-NN
% the prediction from the k-nearest neighbors
function knn_pred = knn_regression(K, D, X)
    x = X(:, 1:end-1);
    k_nn = k_nearest(K, D, x);
    y_i  = k_nn(:,end);
    knn_pred = (1 / K) * (sum(y_i));
    % act = X(:, end)
end

% get a submatrix of the k-nearest neighbors
% get a submatrix of the k-nearest neighbors
function k_nn = k_nearest(K, D, X) 
    [~, idx] = nearest_neighbors(D, X);
    rows    = idx(1:K);
    k_nn    = D(rows, :);
end

% sort the 'train_data' by the euclidean distance from point 'p'
function [nn, i] = nearest_neighbors(train_data, p)
    % input data is all but last column
    [~, columns] = size(train_data);
    feature_len = 1:columns-1;
    
    % sort the rows by the calculated distance
    [nn, i] = sort(vecnorm(train_data(:, feature_len) - p, 2, 2), 'ascend');
end



% b) Linear Regression
function linreg_pred = linreg_regression(D, X)
    x = X(:, 1:end-1);
    
    % f(x) = x * B
    linreg_pred = x * beta_hat(D);
    % act = X(:, end);
end

function B = beta_hat(D) 
    X = D(:, 1:end-1);
    y = D(:, end);
    
    % B = (X^T * X)^-1 * X^T * y
    B = (transpose(X) * X) \ transpose(X) * y;
end



% c) Error Analysis
function R = ps1problem1(train, test)
    % add ones to inputs for bias term
    train = horzcat(ones([size(train,1),1]), train);
    test  = horzcat(ones([size(test,1),1]), test);
    
    K = 5;
    
    rows = size(test, 1);
    knn_pred    = [];
    linreg_pred = [];
    
    % make predictions on test data using k_nn and linreg models
    % collect them in a column vector
    for i = 1:rows
        knn_p    = knn_regression(K, train, test(i, :));
        knn_pred = [knn_pred; knn_p];
        
        linreg_p = linreg_regression(train, test(i, :));
        linreg_pred = [linreg_pred; linreg_p];
    end
    
    % calculate error ratio
    actual  = test(:, end);
    knn_err    = avg_l2_loss(actual, knn_pred);
    linreg_err = avg_l2_loss(actual, linreg_pred);
    R = knn_err / linreg_err;
end

function err = avg_l2_loss(actual, predicted)
    n = size(predicted, 1);
    err = (1/n) * sum( (actual - predicted).^2 );
end