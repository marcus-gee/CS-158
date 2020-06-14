% 3) Best Subset Selection
% a) BSS with the Validation Set Approach

% data
data = table2array(readtable('data/prostate_cancer.csv', 'HeaderLines',1));
ps3problem3a(data);

% inputs included in best model of size p:
% p = 0: 0 
% p = 1: 0     1     
% p = 2: 0     1     2    
% p = 3: 0     1     2     5  
% p = 4: 0     1     2     4     5   
% p = 5: 0     1     2     4     5     8    
% p = 6: 0     1     2     4     5     6     8    
% p = 7: 0     1     2     3     4     5     6     8   
% p = 8: 0     1     2     3     4     5     6     7     8

% BEST MODEL: p_tilde = 3
% Y_hat = f(X) = B0 + B1*X1 + B2*X2 + B5*X5
% Y_hat = f(X) = 2.4694 + 0.6097*X1 + 0.3140*X2 + 0.2215*X5

function ps3problem3a(data) 
    X = data(:, 1:end-2);
    % standardize data
    X = standardize(X);
    % and add ones
    N = length(X);
    X = [ones([N, 1]), X];
    
    Y = data(:, end-1);
    indicator = data(:, end);
    
    % get training and test sets
    [train_x, train_y, test_x, test_y] = split_data(X, Y, indicator);
   
    % collection of the params used in the best model for each p_tilde
    M_star = [];
    
    p = size(X, 2) - 1; % (= 8)
    figure('Name','RSS v. Subset Size')
    for p_tilde = 0:p
        M = nchoosek(p, p_tilde);
        params = [zeros([M, 1]), nchoosek(1:p, p_tilde)];
        
        % fit all M models
        all_err = [];
        for i = 1:M
            xi = train_x(:, (params(i, :) + 1)); % adjust for 1-based indices
            B  = beta_hat(xi, train_y);
            % ccompute the RSS of the fitted model
            rss = RSS(B, xi, train_y);
            all_err = [all_err, rss];
        end
        % define the best model
        [~, best_ind] = min(all_err);
        m_star = params(best_ind, :);
        % formatting for ouput
        m_star = [m_star, zeros([1, p-(p_tilde)])];
        M_star = [M_star; m_star];
        
        % plot all errors for the subsets
        x_plot = p_tilde * ones([1, M]);
        figure(1);
        scatter(x_plot, all_err, 'black');
        hold on
    end
    title("RSS Error v Number of Parameters");
    ylabel("RSS");
    xlabel("Subset Size '$\widetilde{p}$","Interpreter","Latex");
    
    % plot curve of lowest errors for each subset
    lowest_err = [];
    for p_tilde = 0:p
        params = M_star((p_tilde+1), 1:(p_tilde+1));
        xi = train_x(:, params+1); % adjust for 1-based indices
        B  = beta_hat(xi, train_y);
        % ccompute the RSS of the fitted model
        err = RSS(B, xi, train_y);
        lowest_err = [lowest_err, err];
    end
    scatter(0:8, lowest_err, 'red', 'filled');
    plot(0:8, lowest_err, 'red');
    hold off
    
    
    % find best final model
    % plot curve of lowest errors for each subset
    best_err = [];
    for p_tilde = 0:p
        params = M_star((p_tilde+1), 1:(p_tilde+1));
        xi = test_x(:, params+1); % adjust for 1-based indices
        B  = beta_hat(train_x(:, params+1), train_y);
        % compute the RSS of the fitted model
        err = RSS(B, xi, test_y);
        best_err = [best_err, err];
    end
    n = length(test_y);
    best_err = best_err / n;
    
    figure('Name','Test Error v. Subset Size')
    hold on
        scatter(0:8, best_err, 'red', 'filled');
        plot(0:8, best_err, 'red');
        title("Test Error v Number of Parameters");
        ylabel("Test Error");
        xlabel("Model Size '$\widetilde{p}$","Interpreter","Latex");
    hold off
    
    % BEST MODEL:
    % Y_hat = f(X) = B0 + B1*X1 + B2*X2 + B5*X5 
    
    % best parameters for each subset size
    p_tilde = 3+1;
    best_params = M_star(p_tilde, 1:p_tilde);
    
    % best overall model is the full linear regression
    % fit it to training data
    B = beta_hat(train_x(:, best_params+1), train_y)
end

function s = standardize(inputs)
    s = (inputs - mean(inputs)) ./ std(inputs,1);
end

function B = beta_hat(X, y)
    % B = (X^T * X)^-1 * X^T * y
    B = (transpose(X) * X) \ transpose(X) * y;
end

function rss = RSS(B, x, y)
    rss = sum( (y - (x*B)).^2 );
end

function [train_x, train_y, test_x, test_y] = split_data(x, y, ind)
    % find which rows correspond to which dataset
    train_idx = (ind == 1);
    test_idx  = (ind == 0);
    % split dataset in train and test
    train_x = x(train_idx,:);
    train_y = y(train_idx,:);
    
    test_x  = x(test_idx,:);
    test_y = y(test_idx,:);
end