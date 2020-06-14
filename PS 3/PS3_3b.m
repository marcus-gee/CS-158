% 3) Best Subset Selection
% b) BSS with the Cross-Validation and One-Standard-Error Rule

% data
data = table2array(readtable('data/prostate_cancer.csv', 'HeaderLines',1));
ps3problem3b(data);

%{ BEST PARAMETERS/ MODEL:
%  p_tilde_min -> 7
%  p_tilde_star -> 3
%  Y_hat = f(X) = B0 + B1*X1 + B2*X2 + B5*X5 
%  Y_hat = f(X) = 2.4784 + 0.6166*X1 + 0.2820*X2 + 0.2742*X5 
%}

function ps3problem3b(data) 
    X = data(:, 1:end-2);
    % standardize data
    X = standardize(X);
    % and add ones
    N = length(X);
    X = [ones([N, 1]), X];
    Y = data(:, end-1);
    
    R = 100;
    CV_kr = [];
    for r = 1:R
        % (1)
        % make folds 
        splits = randperm(N);
        % indices of where splits happen
        k_folds = [0 19 38 57 77 97];
        
        % cross validation
        K = length(k_folds);
        % used to add errors for each model
        CV_k = zeros([1, 9]);
        % (2)
        for k = 1:(K-1)
            % indices of valdation set 
            valid_ind = splits(:, k_folds(k)+1 : k_folds(k+1));
            train_ind = setdiff(splits, valid_ind);
            
            train_x = X(train_ind, :);
            train_y = Y(train_ind, :);
            valid_x = X(valid_ind, :);
            valid_y = Y(valid_ind, :);
            
            M_star = [];
            % collection of the params used in the best model for each p_tilde
            p = size(X, 2) - 1; % (= 8)
            for p_tilde = 0:p
                M = nchoosek(p, p_tilde);
                params = [zeros([M, 1]), nchoosek(1:p, p_tilde)];

                % (a)
                % fit all M models
                all_err = [];
                for i = 1:M
                    xi = train_x(:, (params(i, :) + 1)); % adjust for 1-based indices
                    B  = beta_hat(xi, train_y);
                    % (b)
                    % compute the RSS of the fitted model
                    rss = RSS(B, xi, train_y);
                    all_err = [all_err, rss];
                end
                % (c)
                % define the best model
                [~, best_ind] = min(all_err);
                m_star = params(best_ind, :);
                % validation error for best model
                vk = length(valid_ind);
                xi = train_x(:, m_star+1);
                % (d)
                B  = beta_hat(xi, train_y);
                Err_kp = (1/vk) * RSS(B, valid_x(:,m_star+1), valid_y);
                % accumulate error
                CV_k(1,p_tilde+1) = CV_k(1,p_tilde+1) + Err_kp;
            end
        end
        % (3)
        % take the average error over each fold
        CV_k = CV_k ./ K;
        % (I)
        CV_kr = [CV_kr; CV_k];
    end
    % (II)
    % compute mean and standard error
    mean_cv = mean(CV_kr);
    se = sqrt(sum((CV_kr - mean_cv).^2) / size(CV_kr, 1));
    x_plot = 0:8;
    
    figure('Name','Test Err v. Number of Params');
    scatter(x_plot,mean_cv,'red','filled');
    hold on
        e = errorbar(x_plot,mean_cv,se,'vertical');
        e.LineStyle = 'none';
        e.Color = 'black';
        title("CV Test Error v Number of Parameters");
        ylabel("CV Test Error");
        xlabel("Model Size '$\widetilde{p}$","Interpreter","Latex");
    hold off
    
    % (III)
    % one standard error rule
    [v, p_tilde_min] = min(mean_cv);
    p_tilde_min = p_tilde_min - 1 % p_tilde_min -> 7
    % (IV)
    p_tilde_star = find(mean_cv < (v + se(p_tilde_min+1)), 1, 'first')-1
    % p_tilde_star -> 3
    
    % BSS with full data on best model 
    M = nchoosek(p, p_tilde_star);
    params = [zeros([M, 1]), nchoosek(1:p, p_tilde_star)];
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
    
    % BEST MODEL:
    % Y_hat = f(X) = B0 + B1*X1 + B2*X2 + B5*X5 

    % fit it to training data
    B = beta_hat(X(:, m_star+1), Y)
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
