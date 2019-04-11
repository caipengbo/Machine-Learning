function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
	g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

	% 使用循环操作向量中的每一个元素
	% for i = [1: length(z)]
	% 	g(i) = 1 / (1 + exp(-z(i)));
	% end

	g = 1 ./ (1 + exp(-z));   %点运算


% =============================================================

end
