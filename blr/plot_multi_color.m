function h1               	= plot_multi_color(name, data, indeces, plotColor)

addpath '../utils';

superLabel                 	= '';

% figureNum                       = 1;
% if ishandle(figureNum)
%     close(figureNum);
% end
% figure(figureNum);

assert(length(data) == 1);


data_i                      = data(1).out;

train_cell                  = data_i.targets_cell(indeces);
test_cell                   = data_i.targetsTest_cell(indeces);
design_train                = data_i.designMat_cell(indeces);
design_test                 = data_i.designMatTest_cell(indeces);

%*****************************
nTasks                      = length(train_cell);

h1                          = [];
hold on;
for j = 1:nTasks

    data_j                  = [train_cell{j};       test_cell{j}];
    design_j                = [design_train{j};     design_test{j}];

    h                       = plot(design_j(:, 2), data_j,  'Marker', '.', 'Color', plotColor,  ...
                                                            'MarkerSize', 12, 'DisplayName', name); %' plotColor '.']);
	set(gca, 'FontSize', 15);
    if j == 1
        h1                  = h;
    end
end

hold off;
%*****************************

   
