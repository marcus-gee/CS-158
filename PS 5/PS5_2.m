% 2) Soft Margin Hyperplane

data = table2array(readtable('data/dataset8.csv',"HeaderLines",1));
ps5problem2(data);

function ps5problem2(D)
    G_minus = D(D(:,end) == -1, :);
    G_plus  = D(D(:,end) == 1, :);
        
    X = D(:, 1:2);
    y = D(:, end);
    
    N = length(X);
    C = [0.1 10];
    i = 1;
    for c = C  
        if c == 0.1
            figure('Name','Soft Margin Hyperplane (C=0.1)')
        else
            figure('Name','Soft Margin Hyperplane (C=10.0)')
        end
        hold on 
            scatter(G_minus(:,1), G_minus(:,2), 'blue');
            scatter(G_plus(:,1), G_plus(:,2), 'red');

            dual(X, y, N, c);

            str = sprintf('Soft Margin Classifier (C = %0.1f)', c);
            title(str);
            xlabel('X1');
            ylabel('X2');
            
            legend('G_-', 'G_+','Support Vectors','H_\beta', ...
                   'H_\beta^-', 'H_\beta^+');
        hold off
        i = i+1;
    end
end

function dual(X, y, N, c)
    H = (y * y') .* (X * X'); 
    f = ones([N,1]);
    Aeq = y';
    beq = 0;
    lb  = zeros([N,1]);
    ub  = 1/c*ones([N,1]);
    lambda = quadprog(H, -f, [], [], Aeq, beq, lb, ub);
    
    % small lambdas -> 0
    lambda(abs(lambda) < 10^-5) = 0;
    
    beta  = (lambda' * (y.*X));
    beta0 = (1/2)*(min(beta*X(y == 1, :)') + max(beta*X(y == -1, :)'));

    beta  = [beta0, beta];
    
    supp_vecs = X(abs(lambda) > 10^-5, :); 
    sprintf('Number of Support Vectors (C= %0.1f): %d', ...
            c, length(supp_vecs))

    % the maximal hyperplane is: 
    %   f(X) = b0 + b1*x1 + b2*x2 = 0
    %   x2 = (b0 + b1*x1) / b2
    function [lower, x2, upper] = plane(b, x1)
        b0 = b(1);
        b1 = b(2);
        b2 = b(3);
        
        x2 = (b0 + b1*x1) / -b2;
        lower = (b0 + b1*x1 + 1) ./ -b2;
        upper = (b0 + b1*x1 - 1) ./ -b2;
    end

    % extract wanted value from plane function
    function lp = lower_plane(b, x1)
        [lp, ~, ~] = plane(b, x1);
    end
    function m = margin(b, x1)
            [~, m, ~] = plane(b, x1);
    end

    function up = upper_plane(b, x1)
            [~, ~, up] = plane(b, x1);
    end

    hold on 
        scatter(supp_vecs(:,1), supp_vecs(:,2), 'xblack')
        fplot(@(x) margin(beta, x), [-4 4], 'black');
        fplot(@(x) lower_plane(beta, x), [-4 4], '--b');
        fplot(@(x) upper_plane(beta, x), [-4 4], '--r');

    hold off
end