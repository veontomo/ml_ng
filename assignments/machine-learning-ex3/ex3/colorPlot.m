%%% http://blogs.mathworks.com/pick/2008/08/15/colors-for-your-multi-line-plots/

data = Theta1;
rows = size(data, 1);
colorSet = varycolor(rows);
set(gca, 'ColorOrder', colorSet);
plot(1:rows, data)
