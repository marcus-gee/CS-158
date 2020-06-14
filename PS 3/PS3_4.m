% 4) Ridge Regression
% data
data = table2array(readtable('data/prostate_cancer.csv', 'HeaderLines',1));
ps3problem4(data);

%{ BEST PARAMETERS/MODEL
%  lambda_min ~= 6 
%  lambda_star ~= ~27
%  Y_hat = f(X) = y_bar + B1*X1 + B2*X2 + B3*X3 + B4*X4 + B5*X5 + B6*X6 + 
%                                                           B7*X7 + B8*X8
%  Y_hat = f(X) = 0 + 0.39*X1 + 0.20*X2 - 0.06*X3 + 0.08*X4 + 0.21*X5 + 
%                                            0.04*X6 +  0.04*X7 + 0.06*X8 
%}

function ps3problem4(data) 
    X = data(:, 1:end-2);
    % and add ones
    N = length(X);

    Y = data(:, end-1);
    %{
    X = (X - mean(X)) ./ std(X,1);
    Y = Y - mean(Y);
    %}
    % number of trials
    R = 100;
    % number of models
    M = 51;
    
    CV_err = zeros([R,M]); 
    for r = 1:R
        % (2)
        % make folds 
        splits = randperm(N);
        % indices of where splits happen
        k_folds = [0 19 38 57 77 97];
        
        % (3)
        % cross validation
        K = length(k_folds);
        for k = 1:(K-1)
            % (a)
            for lambda = 0:(M-1)
                % indices of valdation set 
                valid_ind = splits(:, k_folds(k)+1 : k_folds(k+1));
                train_ind = setdiff(splits, valid_ind);

                train_x = X(train_ind, :);
                train_y = Y(train_ind, :);
                valid_x = X(valid_ind, :);
                valid_y = Y(valid_ind, :);
                
                
                % normalize and center data
                train_x = (train_x - mean(X)) ./ std(X,1);
                train_y = (train_y - mean(Y));
                valid_x = (valid_x - mean(X)) ./ std(X,1);
                valid_y = (valid_y - mean(Y));
                
                
                % (b) fit the model 
                B = beta_hat_R(train_x, train_y, lambda);
                % (c) compute the test error
                vk = length(valid_ind);
                Err_km = (1/vk) * RSS(B, valid_x, valid_y);
                % accumulate error from each fold
                CV_err(r, lambda+1) = CV_err(r, lambda+1) + Err_km;
            end
        end
    end
    % (4) take the average error over each fold
    CV_err = CV_err ./ (K-1);
    % compute mean and standard error
    mean_cv = mean(CV_err);
    se = sqrt(sum((CV_err - mean_cv).^2) / size(CV_err, 1));
    x_plot = 0:(M-1);
    % plot first figure
    figure('Name','Test Err v. Tuning Param')
    scatter(x_plot,mean_cv,'red','filled');
    hold on
        e = errorbar(x_plot,mean_cv,se,'vertical');
        e.LineStyle = 'none';
        e.Color = 'black';
        title('CV Test Error v Tuning Parameter \lambda','Interpreter','tex');
        ylabel('CV Test Error');
        xlabel('Tuning Parameter \lambda','Interpreter','tex');
    hold off
    
    % lambda_min ~= 6 
    % lambda_star ~= 27
     % one standard error rule
    [v, lambda_min] = min(mean_cv);
    lambda_min  = lambda_min - 1
    % find largest lambda with 1-se of smallest cv 
    lambda_star = find(mean_cv < (v + se(lambda_min+1)), 1, 'last') - 1
    
    % fit full model with lambda_star
    X = (X - mean(X)) ./ std(X,1);
    Y = (Y - mean(Y)) ./ std(Y,1);
    B_lambda_star = beta_hat_R(X, Y, lambda_star);
   
    % compute the betas for all lambdas
    B_hat_R = zeros([M, size(X,2)]); 
    for lambda = 0:(M-1)
        % fit the model 
        B_hat_R(lambda+1, :) = beta_hat_R(X, Y, lambda)';
    end
    
    % plot second figure
    figure('Name','Ridge Est v. Tuning Param')
    hold on
    x_plot = 0:(M-1);
    for p = 1:8
        % fit the model 
        plot(x_plot,B_hat_R(:, p));
    end
    xline(lambda_star,'--r');
    scatter(lambda_star*ones([1, 8]),B_lambda_star,'black','filled');
    
    legend({'p';'$\hat{p}^{CS}$'});

    title("Ridge Estimates \beta^R v Tuning Parameter \lambda");
    ylabel("Ridge Estimates \beta^R");
    xlabel('Tuning Parameter \lambda');
    legend('\beta_1','\beta_2','\beta_3','\beta_4','\beta_5','\beta_6','\beta_7','\beta_8')
    hold off    
end

function B = beta_hat_R(X, y, lambda)
    [~, p] = size(X);
    % B = ((X^T * X) + (,\ * Ip))^-1 * X^T * y
    B = ((transpose(X) * X) + (lambda * eye(p))) \ transpose(X) * y;
end


function rss = RSS(B, x, y)
    rss = sum( (y - (x*B)).^2 );
end
