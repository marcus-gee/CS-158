n = 1000;
es = [];
for i = 1:n
    [Ip, X_i, a, pr, e] = expected(50);
    es = [es, e];
end

scatter(1:n, es)
hold on
plot(1:n, ones([1,n]))
hold off
function [Ip, X_i, a, pr, e] = expected(p) 
    N = 100;
    Ip = eye(p);
    X_i = normrnd(0, Ip);
    a = (X_i / norm(X_i, 'fro'));
    pr = (transpose(a) * X_i) * a;
    e = norm(pr, 'fro')^2;
end