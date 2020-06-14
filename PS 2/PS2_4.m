% 2) Linear Regression Analysis of the Prostate Cancer Data.

% data
data = table2array(readtable('data/prostate_cancer.csv', 'HeaderLines',1));
ps2problem4(data);


function ps2problem4(data) 
    input  = data(:, 1:end-2);
    output = data(:, end-1);
    indicator = data(:, end);
    % normalize data
    input = normalize(input);
    [train, test] = split_data(input, output, indicator);
    % add ones to inputs for bias term
    train = horzcat(ones([size(train,1),1]), train);
    test  = horzcat(ones([size(test,1),1]), test);
    
    
    % (a)
    % OLS estimate
    B = beta_hat(train);
    
    
    % (b)
    X      = train(:, 1:end-1);
    y      = train(:, end);
    [N, p] = size(train);
    p      = p-2;
    alpha  = 0.05;
    % calcluate values for table
    y_hat   = X * B;
    sig_hat = sqrt((1 / (N-p-1)) * sum( (y - y_hat).^2 ));
    % vj    = (((X^T)X)-1)_jj
    vj      = diag(inv(X' * X));
    % zj    = B_j / (sig * sqrt(v_j))
    zj      = B ./ (sig_hat * ((vj).^(1/2)));
    % p-values
    mu  = 0;
    sig = 1;
    % Wald test p-value
    pjW = 2 * normcdf((-1 .* abs(zj)), mu, sig);
    % t-tes p-value
    pjF = 2 * tcdf((-1 .* abs(zj)), (N-p-1));
    % confidence interval 
    % ci  = [B +/- z_x/2 * sig_hat * v_j]
    z_o_v = -norminv(alpha/2) * (sig_hat * ((vj).^(1/2)));
    I     = [B - z_o_v, B + z_o_v];
    % show Table
    res = round(horzcat(B, zj, pjW, pjF, I), 4);
    rowNames = {'1','lcavol','lweight','age','lbhp','svi','lcp','gleason','pgg45'};
    colNames = {'B','Zscore','pW','pT','LowerInterval','UpperInterval'};
    Table_ab = array2table(res,'RowNames',rowNames,'VariableNames',colNames)
    
    
    % (c)
    % F-test to test the hypothesis:
    % H0:(B_j4,B_j7,B_j8,B_j9) = (0,0,0,0) v H1:(B_j4,B_j7,B_j8,B_j9) != 0
    
    % beta params to keep
    params   = [1 2 3 5 6];
    train_red= horzcat(train(:, params), train(:, end));
    B_red    = beta_hat(train_red);
    p_red    = 4;
    RSS_Bred = RSS(B_red, train(:, params), y);
    RSS_Bhat = RSS(B, X, y);
    
    f = ((RSS_Bred - RSS_Bhat) / (p - p_red)) / (RSS_Bhat / (N-p-1));
    p_Ftest = 1 - fcdf(f, (p - p_red), (N-p-1))
    
    % (d) 
    n = size(test, 1);
    X = test(:, 1:end-1);
    y = test(:, end);
    % 1. Base model- prediction is mean training value
    base_pred = mean( train(:, end) );
    base_err  = (1/n) * sum( (y - base_pred).^2 );
    % 2. Full model- prediction made using all params
    full_pred = X * B;
    full_err  = (1/n) * sum( (y - full_pred).^2 );
    % 3. Reduced model- prediction made using 
    red_pred = X(:, params) * B_red;
    red_err  = (1/n) * sum( (y - red_pred).^2 );
    % show table of results
    rowNames = {'BaseModel', 'FullModel', 'ReducedModel'};
    colNames = {'Test Error'};
    Table_d = table(base_err, full_err, red_err, 'RowNames',colNames,'VariableNames',rowNames)

end

function [train, test] = split_data(in, out, ind)
    % find which rows correspond to which dataset
    train_idx = ind == 1;
    test_idx  = ind == 0;
    % split dataset in train and test
    train_in  = in(train_idx,:);
    train_out = out(train_idx,:);
    train     = horzcat(train_in, train_out);
    
    test_in  = in(test_idx,:);
    test_out = out(test_idx,:);
    test     = horzcat(test_in, test_out);
end

function B = beta_hat(D) 
    X = D(:, 1:end-1);
    y = D(:, end);
    
    % B = (X^T * X)^-1 * X^T * y
    B = (transpose(X) * X) \ transpose(X) * y;
end

function rss = RSS(B, x, y)
    rss = sum( (y - (x*B)).^2 );
end