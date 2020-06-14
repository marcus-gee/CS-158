% 2a) The Curse of Dimensionality: Theory.

% plot the E[D] vs the dimension for different values of N
figure('Name','The Curse of Dimensionality: Theory')
for n = 2:4
   N = 10 ^n;
   P = 1000;
   fplot(@(p) expected_D(N, p), [1 P]);
   hold on
end
legend('N^2', 'N^3', 'N^4')
title('$\bar{D}$ vs. Dimension','Interpreter','Latex')
ylabel('$\bar{D}$','Interpreter','Latex')
xlabel('Dimension p','Interpreter','Latex')
hold off

% derived expression for the expected distance
% E[D] = (1 / p) * Beta(N+1, 1/p)
function d = expected_D(N, p)
    d = (1 ./ p) .* arrayfun(@(x) beta(N + 1, 1/x), p);
end