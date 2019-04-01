function [vtxs,nvtxs] = fs_read_label(labelf)
% [vtxs,nvtxs] = fs_read_label(labelf)
% 
% Reads the indices of the vertices of a Freesurfer's label. 
%
% Input
% labelf: A Freesurfer's label file.
%
% Output
% vtxs: Indices of the vertices (1-based). 
% nvtxs: The number of vertices in the label.
%
% $Revision: 1.1.1.1 $  $Date: 2012/02/02 11:25:52 $
% Original Author: Jorge Luis Bernal Rusiel 
% CVS Revision Info:
%    $Author: jbernal$
%    $Date: 2012/02/02 11:25:52 $
%    $Revision: 1.1 $
%
fid = fopen(labelf);
tline = fgetl(fid);
nvtxs = fscanf(fid,'%d',1);
vtxs = uint32(fscanf(fid,'%d %*g %*g %*g %*g',[1 nvtxs])); 
vtxs = vtxs + 1;
fclose(fid); 

