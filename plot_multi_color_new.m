function h1               	= plot_multi_color_new(name, dataTrain, dataTest, indeces, plotColor)

addpath '../utils';

superLabel                 	= '';

train_cell                  = dataTrain.targets_cell(indeces);
test_cell                   = dataTest.targets_cell(indeces);
design_train                = dataTrain.designMat_cell(indeces);
design_test                 = dataTest.designMat_cell(indeces);

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

   
