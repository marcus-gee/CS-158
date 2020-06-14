% 2) Decision Boundary of the Bayes Classifier
% b.

ps4problem2b();

function ps4problem2b()
    % constants/parametes
    p = 2;
    m = 5;
    n = 100;
    
    % data
    u = [[-2; 0] [-1; 1] [0; 2] [1; 1] [2; 0]];
    v = [[0; 1] [-1; 0] [0; 0] [1; 0] [0; -1]];
    for i = 0:2
        s = 10^(i-2);
        sig = s * eye(p);
        
        % picking a u_k, v_k
        ui_obs = randi([1 5], 1, n);
        vi_obs = randi([1 5], 1, n);
        
        % observations
        u_obs = (mvnrnd((u(:,ui_obs))', sig))';
        v_obs = (mvnrnd((v(:,vi_obs))', sig))';
        
        figure(i+1)
        % plot 
        hold on
            scatter(u_obs(1,:), u_obs(2,:), 30, 'blue')
            scatter(v_obs(1,:), v_obs(2,:), 30, 'red')
           
            % from Piazza
            % your expression that defines the boundary (found in 2a)
            f=@(X1,X2)  F(X1, X2, u, v, m, s);  
            [X,Y]=meshgrid(-5:0.01:5);
            Z=f(X,Y);
            contour(X,Y,Z, [0,0])
            axis equal
            
            xlabel("X1");
            ylabel("X2");
            str= sprintf("Bayes Classifier Decision Boundary (s=%0.2f)", s);
            title(str);
            legend("G_0", "G_1", "Decision Boundary")
        hold off
    end
end

function output = F(X1, X2, u, v, m, s)
    num_sum   = 0;
    denom_sum = 0;
    for i = 1:m
        num_sum   = num_sum + exp((-1/(2*s))*((X1-u(1, i)).^2 + (X2-u(2, i)).^2));
        denom_sum = denom_sum + exp((-1/(2*s))*((X1-v(1, i)).^2 + (X2-v(2, i)).^2));
    end
    output = (num_sum ./ denom_sum) - 1;
end
