% 5) Training vs Testing Error and The Bias-Variance Trade-Off.
ps1problem5()

function ps1problem5()
    % a) 
    N     = 10 ^3;
    K_max = N;
    
    % err_k -> training error | Err_k -> test error
    err_k = [];
    Err_k = [];
    
    % datasets
    D_train = generate_data(N);
    D_test  = generate_data(N);
    
    train_actual = D_train(:, end);
    test_actual  = D_test(:, end);
    samples = size(D_test, 1);
    
    for k = 1:K_max
        
        train_pred = [];
        test_pred  = [];
        % make predictions on test data using k_nn and linreg models
        % collect them in a column vector
        
        for i = 1:samples  
            % k_nn regression
            train_p = knn_regression(k, D_train, D_train(i, :));
            test_p  = knn_regression(k, D_train, D_test(i, :));
            
            train_pred = [train_pred; train_p];
            test_pred  = [test_pred; test_p];
        end
        
        size(train_pred);
        size(test_pred);
        % calc train/test error
        train_err = avg_l2_loss(train_actual, train_pred);
        err_k = [err_k; train_err];
        
        test_err  = avg_l2_loss(test_actual, test_pred);
        Err_k = [Err_k; test_err];
    end
    
    % plot data
    x = N ./ (1:N);
    figure('Name', 'Error versus Flexibility (a)')
    plot(x, err_k)
    
    hold on
        plot(x, Err_k)
        title("Error versus Flexibility")
        ylabel("Error")
        xlabel("Flexibility (N/k)")
        legend('$\bar{err_k}$','$\bar{Err_k}$', 'Interpreter','Latex')
    hold off
    
    
    % b) 
    T     = 10 ^3;
    N     = 10 ^2;
    K_max = N;
    
    % generate T datasets of N points
    x0 = N .* ones([T, 1]);
    D_train = arrayfun(@(x) generate_data(x), x0,'UniformOutput', false);
    D_test  = generate_data(1);
    
    Err_k = [];
    B2    = [];
    V     = [];
    for k = 1:K_max
        % calc inner sum for variance
        inner_sum = 0;
        for s = 1:T
            Dtrain_i = cell2mat(D_train(s));
            knn_pred = knn_regression(k, Dtrain_i, D_test);
            inner_sum = inner_sum + knn_pred;
        end
        
        % calc errors
        err_sum   = 0;
        bias2_sum = 0;
        var_sum   = 0;
        
        f = sum( transform_mat(D_test(2:4)) );
        
        for t = 1:T 
            % k_nn regression
            Dtrain_i = cell2mat(D_train(t));
            knn_pred  = knn_regression(k, Dtrain_i, D_test);

            % calc errors            
            e = normrnd(0, 1); 
            err_sum   = err_sum + avg_l2_loss(f + e, knn_pred);
            bias2_sum = bias2_sum + knn_pred;
            var_sum   = var_sum + (knn_pred - (inner_sum / T))^2;
        end
        
        Err_k = [Err_k; (err_sum / T)];
        f  = sum( transform_mat(D_test(2:4)) );
        B2 = [B2; (f - (bias2_sum / T))^2];
        
        V = [V; (var_sum / T)];
    end
    
    % make plots
    figure('Name', 'Error versus Flexibility (b)')
    x = N ./ (1:N);
    plot(x, Err_k)
    hold on
        plot(x, B2)
        plot(x, V)
        plot(x, 1 + B2 + V)
        title("Errors versus Flexibility")
        ylabel("Errors")
        xlabel("Flexibility (N/k)")
        legend('Err_k(X)', 'B[f_k(X)]^2', 'V[f_k(X)]', '\sigma^2 + B[f_k(X)]^2 + V[f_k(X)]')
    hold off
end

% generate dataset according to instructions
% param N -> number of points
function data = generate_data(N)
    mu = 0; 
    sigma = 1;
    % sample N points w/ p params from N(0,1)
    X = normrnd(mu, sigma, [N, 3]);
    x = transform_mat(X);
    % Y = sin(X1) + e^X2 + log(|X3|) + eps
    eps = normrnd(mu, sigma, [N, 1]);
    Y   = sum(x, 2) + eps;
    data = horzcat(ones([N, 1]), X, Y);
end

% x = [sin(X1), e^X2, log(|X3|)]
function x = transform_mat(X)
    x1 = sin(X(:, 1));
    x2 = exp(X(:, 2));
    x3 = log( abs(X(:, 3)) );
    x = horzcat(x1, x2, x3);
end

% error function
function err = avg_l2_loss(actual, predicted)
    n = size(predicted, 1);
    err = (1/n) * sum( (actual - predicted).^2 );
end


% k-NN
% the prediction from the k-nearest neighbors
function knn_pred = knn_regression(K, D, X)
    x = X(:, 1:end-1);
    k_nn = k_nearest(K, D, x);
    y_i  = k_nn(:,end);
    knn_pred = (1 / K) * (sum(y_i));
end

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
