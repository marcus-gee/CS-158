% 1) Maximal Margin Hyperplane

data = table2array(readtable('data/dataset7.csv',"HeaderLines",1));
ps5problem1(data);

function ps5problem1(D)
    G_minus = D(D(:,end) == -1, :);
    G_plus  = D(D(:,end) == 1, :);
    
    X = D(:, 1:2);
    y = D(:, end);
   
    N = 200;
    figure('Name','Maximal Margin Hyperplane')
    hold on 
        scatter(G_minus(:,1), G_minus(:,2), 'blue');
        scatter(G_plus(:,1), G_plus(:,2), 'red');
        
        primal(X, y, N);
        % betas:
        % [B0, B1, B2] = [13.625, -2.727, -3.271]
        
        dual(X, y, N);
        % betas:
        % [B0, B1, B2] = [-13.625, 2.727, 3.271]
        
        title('Maximal Margin Classifier (Primal and Dual)');
        xlabel('X1');
        ylabel('X2');
        legend('G_-', 'G_+', 'Primal', 'Dual', 'Support Vectors');
    hold off
end

function primal(X, y, N)
    X = [ones([N,1]), X];
    H = eye(3);
    H(1,1) = 0;
    f = zeros([3,1]);
    A = y.*X;
    b = -ones([N,1]);
    beta = quadprog(H, f, -A, b);
    
    sprintf('Primal Form:\n[B0, B1, B2] = %0.3f %0.3f %0.3f', beta)
    
    fplot(@(x) plane(beta, x), [-4 8], '--black');
end 

function dual(X, y, N)
    H = (y * y') .* (X * X');
    f = ones([1,N]);
    A = eye(N);
    b = zeros([N,1]);
    Aeq = y';
    beq = 0;
    lambda = quadprog(H, -f, -A, b, Aeq, beq);
    
    % small lambdas -> 0
    lambda(abs(lambda) < 10^-5) = 0;
    
    beta  = (lambda' * (y.*X))';
    beta0 = (-1/2)*(min(beta'*X(y == 1, :)') + max(beta'*X(y == -1, :)'));
    beta  = [beta0; beta];
    sprintf('Dual Form:\n[B0, B1, B2] = %0.3f %0.3f %0.3f', beta)
    
    supp_vecs = X(abs(lambda) > 10^-5, :); 

    hold on
        fplot(@(x) plane(beta, x), [-4 8], ':black');
        scatter(supp_vecs(:,1), supp_vecs(:,2), 'green', 'filled')
    hold off
end

% the maximal hyperplane is: 
%   f(X) = b0 + b1*x1 + b2*x2 = 0
%   x2 = (b0 + b1*x1) / b2
function x2 = plane(b, x1)
    b0 = b(1);
    b1 = b(2);
    b2 = b(3);

    x2 = (b0 + b1*x1) / -b2;
end
