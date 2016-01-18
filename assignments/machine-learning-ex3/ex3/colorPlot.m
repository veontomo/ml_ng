%%% http://blogs.mathworks.com/pick/2008/08/15/colors-for-your-multi-line-plots/

data = prob;
rows = size(data, 1);
cols = size(data, 2);
colorSet = varycolor(rows);
set(gca, 'ColorOrder', colorSet);
plot(1:cols, data(1:20, :), 'x')
