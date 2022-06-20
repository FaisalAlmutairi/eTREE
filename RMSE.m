function [RMSE] = RMSE(x, x_hat, min_x, max_x)

% a helper function to compute the root mean square error 

x_hat(x_hat<=min_x) = min_x;
x_hat(x_hat>=max_x) = max_x;
RMSE = sqrt(mean((x - x_hat).^2));

end