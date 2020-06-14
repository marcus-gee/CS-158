% 4) K-NN and Linear Regression for Classification.

% data
train3 = table2array(readtable('data/dataset3_train.csv', 'HeaderLines',1));
test3  = table2array(readtable('data/dataset3_test.csv', 'HeaderLines',1)); 
train4 = table2array(readtable('data/dataset4_train.csv', 'HeaderLines',1));
test4  = table2array(readtable('data/dataset4_test.csv', 'HeaderLines',1));

% misclassification ratios
R3 = ps1problem4(train3, test3)
% R3 = 1.3806
% bc R1 > 1, the linear regression model outperformed the k_nn model 

R4 = ps1problem4(train4, test4)
% R4 = 0.6892
% bc R2 < 1, the k_nn model outperformed the linear regression model



% a) k-NN classification functions
% the prediction from the k-nearest neighbors
function knn_pred = knn_classification(K, D, X)
    x = X(:, 1:end-1);
    k_nn = k_nearest(K, D, x);
    y_i  = k_nn(:,end);
    % Find P(G = 1 | X = x)
    knn_pred = (1 / K) * (sum(y_i));
    % because k = 1, knn_pred will be 1 G_nn = 1 and 0 otherwise.
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



% b) Linear Regression classification functions
function linreg_pred = linreg_classification(D, X)
    % we have the binary variables Y = 0 for G_0 and Y = 1 for G_1
    % thus, prediction G-hat is g_0 if Y-hat ('linreg_pred') < 1/2, 
    % otherwise, g_1
    x = X(:, 1:end-1);
    
    % Y-hat = f(x) = x * B
    f = x * beta_hat(D);
    if f < 0.5 linreg_pred = 0; else linreg_pred = 1; end
end

function B = beta_hat(D) 
    X = D(:, 1:end-1);
    y = D(:, end);
    
    % B = (X^T * X)^-1 * X^T * y
    B = (transpose(X) * X) \ transpose(X) * y;
end


% c) Misclassification Rates
function R = ps1problem4(train, test)
    % add ones to inputs for bias term
    train = horzcat(ones([size(train,1),1]), train);
    test  = horzcat(ones([size(test,1),1]), test);

    K = 1;
    
    rows = size(test, 1);
    knn_pred    = [];
    linreg_pred = [];
    
    % make predictions on test data using k_nn and linreg models
    % collect them in a column vector
    for i = 1:rows
        knn_p    = knn_classification(K, train, test(i, :));
        knn_pred = [knn_pred; knn_p];
        
        linreg_p = linreg_classification(train, test(i, :));
        linreg_pred = [linreg_pred; linreg_p];
    end
    
    % calculate error ratio
    actual  = test(:, end);
    knn_misclass    = misclass_rate(actual, knn_pred);
    linreg_misclass = misclass_rate(actual, linreg_pred(:, end));
    R = knn_misclass / linreg_misclass;
end

function misclass = misclass_rate(actual, predicted)
    n = size(predicted, 1);
    misclass = (1/n) * sum( actual ~= predicted );
end
