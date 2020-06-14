% 2) Leave-One-Out-Cross-Validation for Model Selection
data = table2array(readtable('data/dataset5.csv', 'HeaderLines',1));

% run main
ps3problem2(data)

%{ based on the bar plot output, the best model to choose is likely model 2
%  since it has lowest average error of the three models.
%  AVERAGE ERRORS -> [1.1075    1.0803    1.5001]
%}
function ps3problem2(data)
    N = length(data);
    % add ones
    data = [ones([N, 1]), data];

    err1 = 0;
    err2 = 0;
    err3 = 0;
    for i = 1:N
        % leave one out
        train_ind = [1:i-1, i+1:N];
        train_set = data(train_ind, :);
        train_x = train_set(:, 1:end-1);
        train_y = train_set(:, end);
        
        valid_ind = i;
        valid_set = data(valid_ind, :);
        valid_x = valid_set(:, 1:end-1);
        valid_y = valid_set(:, end);
        
        train_x1 = train_x;
        train_x2 = [train_x(:, 1:2), sin(train_x(:, 3))];
        train_x3 = train_x(:, 1:2);
        
        valid_x1 = valid_x;
        valid_x2 = [valid_x(:, 1:2), sin(valid_x(:, 3))];
        valid_x3 = valid_x(:, 1:2);
        
        % fit data and get estimates
        B1 = beta_hat(train_x1, train_y);
        B2 = beta_hat(train_x2, train_y);
        B3 = beta_hat(train_x3, train_y);
        
        y1 = f1(B1, valid_x1);
        y2 = f2(B2, valid_x2);
        y3 = f3(B3, valid_x3);
        
        % find errors
        err1 = err1 + ((valid_y - y1).^2);
        err2 = err2 + ((valid_y - y2).^2);
        err3 = err3 + ((valid_y - y3).^2);
       
    end
    Err1 = (1/N) * err1;
    Err2 = (1/N) * err2;
    Err3 = (1/N) * err3;
    Err = [Err1, Err2, Err3]
end

function B = beta_hat(X, y)
    % B = (X^T * X)^-1 * X^T * y
    B = (transpose(X) * X) \ transpose(X) * y;
end
% f1(X) = B0 + B1X1 + B2X2
function y = f1(B, X)
    y = X * B;
end
% f2(X) = B0 + B1X1 + B2sin(X2)
function y = f2(B, X)
    % matlab has 1-based indexing, so indices are i+1
    y = X * B;
end
%f1(X) = B0 + B1X1
function y = f3(B, X)
    y = X * B;
end