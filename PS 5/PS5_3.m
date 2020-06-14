% 3) Support Vector Machine

data = table2array(readtable('data/dataset9.csv',"HeaderLines",1));
ps5problem3(data);

function ps5problem3(D)
    G_minus = D(D(:,end) == -1, :);
    G_plus  = D(D(:,end) == 1, :);
    
    X = D(:, 1:2);
    y = D(:, end);
        
    N = length(X);
    
    figure('Name','Support Vector Machine')
    hold on 
        scatter(G_minus(:,1), G_minus(:,2), 'blue');
        scatter(G_plus(:,1), G_plus(:,2), 'red');
        
        for c = 0:1
            dual(X, y, N, c);
        end
        % 'Number of Support Vectors (C= 0.0): 30'
        % 'Number of Support Vectors (C= 1.0): 61'
        % 
        % I believe the C=1 decision boundary is more reasonable, because
        % because by allowing our model to have some boundary violations,
        % we are making it more flexible. This allows us to better
        % generalize our data and not overfit it. 
        
        title('Support Vector Machine Decision Boundary');
        xlabel('X1');
        ylabel('X2');
        legend('G_-', 'G_+', 'C=0 Decision Boundary', 'C=1 Decision Boundary');
    hold off
end

function dual(X, y, N, c)
    K = gauss_kernel(X, X);
    H = (y * y') .* (K); 
    f = ones([N,1]);
    Aeq = y';
    beq = 0;
    lb  = zeros([N,1]);
    ub  = 1/c*ones([N,1]);
    lambda = quadprog(H, -f, [], [], Aeq, beq, lb, ub);
    
    % support vectors -> xi, where lambda_i > 10^-5
    supp_vecs = X(abs(lambda) > 10^-5, :); 
    supp_ind = abs(lambda) > 10^-5;
    lambda(abs(lambda) < 10^-5) = [];
    
    xs = X(supp_ind,:);
    ys = y(supp_ind);
    
    % support vectors from +1 class
    % guassian kernel between all support vectors and those in +1 class
    xs_plus = xs(ys == 1, :);
    mat_plus = zeros([length(xs), length(xs_plus)]);
    for i = 1:length(xs)
        mat_plus(i,:) = exp(-(vecnorm(xs(i,:) - xs_plus,2,2).^2))';
    end
    
    % support vectors from -1 class
    % guassian kernel between all support vectors and those in -1 class
    xs_min = xs(ys == -1, :); 
    mat_min = zeros([length(xs), length(xs_min)]);
    for i = 1:length(xs)
        mat_min(i,:) = exp(-(vecnorm(xs(i,:) - xs_min,2,2).^2))';
    end
    
    ly = lambda .* ys;
    mat_plus = ly' * mat_plus;
    mat_min = ly' * mat_min;

    % using equation found from bottom right of lecture notes p.84
    beta0 = (-1/2)*(min(mat_plus) + max(mat_min));

    % find values that satisfy margin
    function out = F(X, b0, l, y, s)
        n = length(X);
        out = [];
        for x = 1:n
            xi = X(x,:);
            fx = l' * (y .* (exp(-vecnorm(s - xi, 2, 2).^2)));
            fx = fx + b0;
            
            % values of (x1,x2) that satisfy margin (f(x) = 0)
            eps = 0.04;
            if abs(fx) <= eps
                out = [out; xi];
            end
        end        
    end

    hold on     
        sprintf('Number of Support Vectors (C= %0.1f): %d', ...
            c, length(supp_vecs))
        f=@(X)  F(X, beta0, lambda, ys, supp_vecs);  
        [X,Y]=meshgrid(-2.5:0.004:2.5);
        input = [X(:), Y(:)];
        out =f(input);
        scatter(out(:,1), out(:,2), 10,'.');
    hold off
end

function out = gauss_kernel(Xi, Xj)
    Ni = length(Xi);
    Nj = length(Xj);
    out = zeros([Ni,Nj]);
    
    for i = 1:Ni
        xi = Xi(i,:);
        for j = 1:Nj
            xj = Xj(j,:);
            out(i,j) = exp(- norm(xi - xj)^2 );
        end
    end
end