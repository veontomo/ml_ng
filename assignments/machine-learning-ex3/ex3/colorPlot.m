data = Theta1;
rows = size(data, 1);
colorSet = varycolor(rows);
set(gca, 'ColorOrder', colorSet);
plot(1:rows, data)
