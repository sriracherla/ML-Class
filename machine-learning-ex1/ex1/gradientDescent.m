function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    % h  = X * theta;
    % h_minus_y = h - y;
    % sum_h_minus_y = sum(h_minus_y);

    % h_minus_y_times_x = h_minus_y .* X(:,2); 
    % we use X(:,2) since we are computing t1 (remember in formula it starts at zero and in Octave it starts at 1)
    % remember it is elementwise multiplication 

    % sumh_minus_y_times_x = sum(h_minus_y_times_x);

    % t0 = theta(1) - (alpha / m) * sum_h_minus_y;
    % t1 = theta(2) - (alpha / m) * sumh_minus_y_times_x;

    % fprintf("computed cost for theta: \n%4.2f %4.2f\n is %0.4f\n", theta);
    % fprintf("\nis %0.4f\n", J_history(iter));

    % theta = [t0; t1];
    % fprintf("\nnew theta: \n%4.2f %4.2f\n is %0.4f\n", theta);

    % optimal solution
    active_theta = theta; % for simultaneous update
    for i = 1: length(theta)
        theta(i) = active_theta(i) - ((alpha / m) * (sum((X * active_theta - y) .* X(:,i))));
    endfor
end

end
