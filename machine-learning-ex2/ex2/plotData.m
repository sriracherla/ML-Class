function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% plot everything
crosses = X(find(y==0),:);
scatter(crosses(:,1), crosses(:,2), 10, "b", "s", "filled");

% override with plot for 1 with a different color
circles = X(find(y==1),:);
scatter(circles(:,1), circles(:,2), 10, "r", "+", "filled");

% xlabel("Exam 1 score");
% ylabel("Exam 2 score");
% title ("Admitted vs. Not admitted based on Exam 1 and Exam 2 scores");
% h = legend ({"Admitted", "Not admitted"});
% legend (h, "location", "northeastoutside");

% =========================================================================

hold off;

end
