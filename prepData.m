function out                    = prepData(ids, inTimes, inMeasures, P, indicesCell)

out.n_tasks                    	= length(indicesCell);

[out.times_cell,        ...
 out.designMat_cell,    ...
 out.targets_cell]            	= deal(cell(out.n_tasks, 1));

out.ids                         = unique(ids, 'stable');
out.nSamples                   	= zeros(out.n_tasks, 1);


for i = 1:out.n_tasks

    index_i                     = indicesCell{i};
    out.nSamples(i)           	= length(index_i);
    
    t_i                         = inTimes(index_i);
    data_i                      = inMeasures(index_i);
    
    Z_i                         = zeros(out.nSamples(i), P+1);
    for j = 0:P
        Z_i(:, j+1)             = t_i .^ j;
    end   
   
    out.times_cell{i}          	= t_i;    
    out.designMat_cell{i}      	= Z_i;     
    out.targets_cell{i}        	= data_i;        
end


