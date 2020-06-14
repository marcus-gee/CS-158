% 1) The LASSO

% data
data = table2array(readtable("data/prostate_cancer.csv", "HeaderLines",1));
ps4problem1(data);
% lambda_min -> ~0.02
% lambda_star -> ~0.011

function ps4problem1(data)
    X = data(:, 1:end-2);
    Y = data(:, end-1);
    N = length(X);

    R = 100;
    M = 1;
    CV_err = zeros([R, (M/0.01)]);
    for r = 1:R
        % (1)
        % make folds 
        splits = randperm(N);
        % indices of where splits happen
        k_folds = [0 19 38 57 77 97];
        
        % cross validation
        K = length(k_folds);

        % (2)
        for k = 1:(K-1)
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
                
            
            % (a)
            % fit all M models
            for i = 0.01:0.01:M
                B  = lasso(train_x,train_y,"Lambda",i,"Standardize",false);
                % (b)
                % compute the RSS of the fitted model
                rss = RSS(B, valid_x, valid_y);
                idx = round(i / 0.01);
                CV_err(r, idx) = CV_err(r, idx) + (rss / K);
            end
        end
    end
    % (II)
    % compute mean and standard error
    mean_cv = mean(CV_err);
    se = sqrt(sum((CV_err - mean_cv).^2) / size(CV_err, 1));
    x_plot = 0.01:0.01:M;
    figure('Name','Test Err v. Tuning Param')
    % TODO: change labels and title
    scatter(x_plot,mean_cv,"red","filled");
    hold on
        e = errorbar(x_plot,mean_cv,se,"vertical");
        e.LineStyle = "none";
        e.Color = "black";
        title("CV Test Error v Tuning Parameter \lambda^*");
        ylabel("Average CV Test Error");
        xlabel("Tuning Parameter \lambda^*");
    hold off
            
    % (III)
    % one standard error rule
    [v, lambda_min] = min(mean_cv);
    lambda_min = (lambda_min - 1); 
    % lambda_min -> ~0.02
    % (IV)
    lambda_star = (find(mean_cv < (v + se(lambda_min+1)), 1, "last")-1);
    % lambda_star -> ~0.11
    
    lambda_min = lambda_min / 100
    lambda_star = lambda_star / 100
    
    p = 8;
    B_hat_R = zeros([(M/0.01), p]);
    for lambda = 0.01:0.01:M
        idx = round(lambda / 0.01);
        B_hat_R(idx, :) = lasso(X,Y,"Lambda",lambda,"Standardize",false);
    end 
    
    % plot second figure
    figure('Name','Lasso Estimate v. Tuning Param')
    hold on
    x_plot = 0.01:0.01:M;
    for p = 1:8
        % fit the model 
        plot(x_plot,B_hat_R(:, p));
    end
    
    % plot
    xline(lambda_star,"--r");
    scatter(lambda_star*ones([1, 8]),B_hat_R(100*lambda_star, :),"black","filled");
    
    title("Lasso Estimates \beta^L v Tuning Parameter \lambda");
    ylabel("Lasso Estimates \beta^L");
    xlabel("Tuning Parameter \lambda");
    legend("\beta_1","\beta_2","\beta_3","\beta_4","\beta_5","\beta_6","\beta_7","\beta_8")
    hold off   
end

function rss = RSS(B, x, y)
    rss = sum( (y - (x*B)).^2 )/length(x);
end
