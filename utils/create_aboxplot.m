%*****************************
function create_aboxplot(X, label_names, fontRotation, fontSize)

addpath '../aboxplot';

aboxplot(X, 'labels', label_names, 'colorgrad','green_down', 'plotMean', false); 
rotateFigFont(fontRotation); 
set(gca, 'FontSize', fontSize);