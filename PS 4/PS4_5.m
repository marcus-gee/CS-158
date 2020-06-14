% 5) Linear and Quadratic Discriminant Analysis

train = table2array(readtable('data/dataset6_train.csv',"HeaderLines",1));
test  = table2array(readtable("data/dataset6_test.csv", "HeaderLines",1));
ps4problem5(train, test);

function ps4problem5(train, test)
    % a. find LDA and QDA decision boundaries with training data
    eps = 10^-1; 
    % train data based on class
    g0 = train(train(:,3) == 0, 1:2);
    g1 = train(train(:,3) == 1, 1:2);
    
    % find boundaries
    dx  = 0.05;
    min = -6;
    max = -1 * min;
        
    % QDA:
    % in order to get a smooth-ish curve for qda i split x1 into neg and pos
    % halves so i could use more points
    % x1 from min to 0
    [x1n, x2] = meshgrid((min:dx:0),(min:dx:max));
    X = [x1n(:), x2(:)];
    % calculate discriminant functions
    [qda_0, qda_1] = qda_delta(g0', g1', [x1n(:), x2(:)]');
    % (x1, x2) pairs where |d_0 - d_1| < e    
    qda_d = abs(qda_0 - qda_1) < eps;
    X_qdan = X(qda_d, :);
    
    % x1 from 0 to max
    [x1p, x2] = meshgrid((0:dx:max),(min:dx:max));
    X = [x1p(:), x2(:)];
    % calculate discriminant functions
    [qda_0, qda_1] = qda_delta(g0', g1', [x1p(:), x2(:)]');
    % (x1, x2) pairs where |d_0 - d_1| < e
    qda_d = abs(qda_0 - qda_1) < eps;
    X_qdap = X(qda_d, :);
    
    % combine results
    X_qda = [X_qdan; X_qdap];
    
    % LDA:
    [x1, x2] = lda_delta(g0', g1', (min:max)');

    figure('Name','LDA and QDA Boundary');
    hold on
        scatter(g0(:,1), g0(:,2), 'red');
        scatter(g1(:,1), g1(:,2), 'blue');
        plot(x1, x2, '-black');
        plot(X_qda(:,1), smooth(X_qda(:,2)), '--black')
        xlabel("x1");
        ylabel("x2");
        title("LDA and QDA Decision Boundaries for dataset 6");
        legend("G_0", "G_1", "LDA", "QDA");
    hold off
    
    
    % b. find average test err. which is classifier is more efficient?
    test_act = test(:, end);
    n = length(test_act);
    
    qda_pred = qda_prediction(g0', g1', test(:, 1:2)');
    lda_pred = lda_prediction(g0', g1', test(:, 1)', test(:, 2)');
    
    qda_err = (1/n) * sum(qda_pred ~= test_act') % = 0.0590
    lda_err = (1/n) * sum(lda_pred ~= test_act') % = 0.1270
    % in this case we can conclude that the QDA classifier is more 
    % efficient, since the test error is smaller.
    
    % c. plot the ROC curves. which is classifier is more efficient?
    alpha = 0:0.01:1;
    Err_0 = zeros([length(alpha), 2]);
    Err_1 = zeros([length(alpha), 2]);
    for a = alpha
        % make predictions
        lda_pred = lda_prediction2(g0', g1', test(:, 1:2)', a);
        % classify correctness/errors of predictions
        n00 = sum((lda_pred == 0) & (test(:, 3)' == 0));
        n01 = sum((lda_pred == 0) & (test(:, 3)' == 1));
        n11 = sum((lda_pred == 1) & (test(:, 3)' == 1));
        n10 = sum((lda_pred == 1) & (test(:, 3)' == 0));
        
        lda_err_0 = n10 / (n00 + n10);
        lda_err_1 = n01 / (n11 + n01);
        
        % now for QDA
        qda_pred = qda_prediction2(g0', g1', test(:, 1:2)', a);
        n00 = sum((qda_pred == 0) & (test(:, 3)' == 0));
        n01 = sum((qda_pred == 0) & (test(:, 3)' == 1));
        n11 = sum((qda_pred == 1) & (test(:, 3)' == 1));
        n10 = sum((qda_pred == 1) & (test(:, 3)' == 0));
        
        qda_err_0 = n10 / (n00 + n10);
        qda_err_1 = n01 / (n11 + n01);

        % store errors
        Err_0(round(a/0.01)+1, :) = [lda_err_0 qda_err_0];
        Err_1(round(a/0.01)+1, :) = [lda_err_1 qda_err_1];
    end
    % plot ROC curve
    figure('Name','ROC Curve');
    hold on
        plot(Err_0(:, 1), (1 - Err_1(:, 1)), "red");
        plot(Err_0(:, 2), (1 - Err_1(:, 2)), "blue");
        xlabel("Err_0(x)");
        ylabel("1 - Err_1(x)");
        title("ROC Curve");
        legend("LDA", "QDA")
    hold off
    
    % classification inequality
    % decision boundary is given by:
    %   P(G=G_0|X=x) = a AND P(G=G_1|X=x) = 1-a
    % this is equivalent to:
    %   P(G=G_0|X=x) = (a/1-a) * P(G=G_1|X=x)
    % taking log of both sides and simplifying in terms of discriminant 
    % functions:
    %   d_0 = log(a/1-a) + d_1
    % so, x is assigned to G_0, if 
    %   d_0 > log(a/1-a) + d_1, 
    % otherwise G_1.
   
    % From looking at the ROC Curves, the QDA classifier method appears 
    % to be more efficient, as the AUC (area under curve is larger for the
    % the QDA ROC curve compared to the LDA one. This means the overall 
    % performance of the QDA is better.
end

% returns the value from the discriminant function for class G_k
function [x1, x2] = lda_delta(g0, g1, x1)
    K = 2;
    [~, N_0] = size(g0);
    [~, N_1] = size(g1);
    N = N_0 + N_1;
    
    % get values needed to calc discriminant function
    mu_0 = (1 / N_0) .* sum(g0, 2);
    pi_0 = N_0 / N;
    
    mu_1 = (1 / N_1) .* sum(g1, 2);
    pi_1 = N_1 / N;
    
    cov = (g0 - mu_0)*(g0 - mu_0)' + (g1 - mu_1)*(g1 - mu_1)';
    cov = (1 / (N-K)) * cov;
       
    lhs = (1/2)*((mu_0' / cov) * mu_0) - (1/2)*((mu_1' / cov) * mu_1) ...
          + log(pi_1) - log(pi_0);
    rhs = ((mu_0' / cov) - (mu_1' / cov));
    x2 = (lhs - rhs(1)*x1) / rhs(2);
end

% LDA predictions for when threshold is 1/2
function pred = lda_prediction(g0, g1, x1, x2_act)
    [~, x2] = lda_delta(g0, g1, x1);
    pred = x2_act > x2;
end

% LDA predictions for a variable threshold
function pred = lda_prediction2(g0, g1, xi, threshold)
    x1 = xi(1, :);
    x2_act = xi(2, :);
    
    K = 2;
    [~, N_0] = size(g0);
    [~, N_1] = size(g1);
    N = N_0 + N_1;
    
    % get values needed to calc discriminant function
    mu_0 = (1 / N_0) .* sum(g0, 2);
    pi_0 = N_0 / N;
    
    mu_1 = (1 / N_1) .* sum(g1, 2);
    pi_1 = N_1 / N;
    
    cov = (g0 - mu_0)*(g0 - mu_0)' + (g1 - mu_1)*(g1 - mu_1)';
    cov = (1 / (N-K)) * cov;
       
    lhs = (1/2)*((mu_0' / cov) * mu_0) - (1/2)*((mu_1' / cov) * mu_1) ...
          + log(pi_1) - log(pi_0) + log(threshold/ (1-threshold));
    rhs = ((mu_0' / cov) - (mu_1' / cov));
    x2 = (lhs - rhs(1)*x1) / rhs(2);
    pred = x2_act > x2; 
end

function [qda_0, qda_1] = qda_delta(g0, g1, xi)
    [~, N_0] = size(g0);
    [~, N_1] = size(g1);
    N = N_0 + N_1;
    
    % get values needed to calc discriminant function
    mu_0 = (1 / N_0) .* sum(g0, 2);
    pi_0 = N_0 / N;
    cov_0= (1 / (N_0-1)) * (g0 - mu_0)*(g0 - mu_0)'; 
    
    mu_1 = (1 / N_1) .* sum(g1, 2);
    pi_1 = N_1 / N;
    cov_1= (1 / (N_1-1)) * (g1 - mu_1)*(g1 - mu_1)';
    
    % calc discriminant function
    qda_0 = (-1/2)*diag(((xi - mu_0)' / cov_0)*(xi - mu_0))' ...
            - (1/2)*(log(det(cov_0))) + log(pi_0);
    qda_1 = (-1/2)*diag(((xi - mu_1)' / cov_1)*(xi - mu_1))' ...
            - (1/2)*(log(det(cov_1))) + log(pi_1);
end

% QDA predictions for when threshold is 1/2
function pred = qda_prediction(g0, g1, xi)
    [qda_0, qda_1] = qda_delta(g0, g1, xi);
    pred = qda_1 > qda_0;
end

% QDA predictions for a variable threshold
function pred = qda_prediction2(g0, g1, xi, threshold)
    [qda_0, qda_1] = qda_delta(g0, g1, xi);
    pred = (qda_1 + log(threshold/ (1-threshold))) > qda_0;
end
