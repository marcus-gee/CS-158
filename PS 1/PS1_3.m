% 3) The Curse of Dimensionality: Simulation.
ps1problem3()

% find the error on k_nn and linreg models for p = 1..100 and plot the 
% findings
function ps1problem3()
    N = 10^3;
    p_max = 100;
    
    knn_errors    = [];
    linreg_errors = [];
    for p = 1:p_max
        train = generate_data(N, p);
        test  = generate_data(N, p);  
        
        [knn_err, linreg_err] = model_errors(train, test);
        knn_errors    = [knn_errors, knn_err];
        linreg_errors = [linreg_errors, linreg_err];
    end
    
    % plot data
    x = 1:p_max;
    figure('Name','The Curse of Dimensionality: Simulation (K-NN)')
    hold on
        plot(x, knn_errors)
        title("K-NN Average Test Error Versus the Dimension p")
        ylabel("K-NN Average Test Error")
        xlabel("Dimension p")
    hold off
    
    figure('Name','The Curse of Dimensionality: Simulation (Lin Reg)')
    hold on
        plot(x, linreg_errors)
        title("Linear Regression Average Test Error Versus the Dimension p")
        ylabel("Linear Regression Average Test Error")
        xlabel("Dimension p")
        sigma = 1;
        theoretical_linreg = (sigma) + (((sigma) / N) * x);
        plot(x, theoretical_linreg)
    hold off
end

% generate dataset according to instructions
% param N -> number of points
function data = generate_data(N, p)
    mu = 0; 
    sigma = 1;
    % sample N points w/ p params from N(0,1)
    X = normrnd(mu, sigma, [N, p]);
    Z = normrnd(mu, sigma, [N, 1]);
    % y = sum of x_i
    y = sum(X, 2) + Z;
    data = horzcat(ones([N, 1]), X, y);
end

% get average error for the k_nn and linreg models from the datasets
function [knn_err, linreg_err] = model_errors(train, test)
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
    
    % calculate error
    actual  = test(:, end);
    knn_err    = avg_l2_loss(actual, knn_pred);
    linreg_err = avg_l2_loss(actual, linreg_pred);
end

function err = avg_l2_loss(actual, predicted)
    n = size(predicted, 1);
    err = (1/n) * sum( (actual - predicted).^2 );
end




% Functions from (1)

% k-NN functions
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



% Linear Regression functions
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