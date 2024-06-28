function mesh = add_cylinder_fl(mesh, cylinder, ptCloud)

% mesh = add_cylinder_fl(mesh, cylinder)
% Adds cylindrical fluorescence anomalies to the mesh.
% 
% Attempts to functions similarly to add_blob_fl
% cylinder.r - radius of cylinder
% cylinder.tform - axis of cylinder
% cylinder.height - height of cylinder
% cylinder.muax - absorption coefficient at excitation
% cylinder.musx - scatter coefficient at excitation
% cylinder.muam - absorption coefficient at emission
% cylinder.musm - scatter coefficient at emission
% cylinder.muaf - absorption of fluorophore
% cylinder.eta - quantum yield of fluorophore
% cylinder.tau - lifetime of fluorophore
% cylinder.ri - refractive index
% cylinder.region - region number
% cylinder.dist - distance between nodes for anomaly (BEM only)




% If not a workspace variable, load mesh
if ischar(mesh)== 1
  mesh = load_mesh(mesh);
end


if (isfield(cylinder, 'x') == 0)
    errordlg('No x coordinate was given for the anomaly','NIRFAST Error');
    error('No x coordinate was given for the anomaly');
end
if (isfield(cylinder, 'y') == 0)
    errordlg('No y coordinate was given for the anomaly','NIRFAST Error');
    error('No y coordinate was given for the anomaly');
end
if (isfield(cylinder, 'z') == 0 || mesh.dimension == 2)
    cylinder.z = 0;
end
if (isfield(cylinder, 'r') == 0)
    errordlg('No radius was given for the anomaly','NIRFAST Error');
    error('No radius was given for the anomaly');
end
if (isfield(cylinder, 'height') == 0)
    errordlg('No height was given for the anomaly','NIRFAST Error');
    error('No height was given for the anomaly');
end

% dist = distance(mesh.nodes(:,1:3),ones(length(mesh.bndvtx),1),[cylinder.x cylinder.y cylinder.z]);


%Rotate the point cloud, equivalent to rotating the cylinder, but the
%function only allows axis aligned cylinders
ptCloudOut = pctransform(ptCloud, cylinder.tform);

indices = find(findPointsInCylinder(ptCloudOut, cylinder.r, Height=cylinder.height, Center=[0,0,0], VerticalAxis='Z')==1);

clear ptCloudOut;

%custom_filter = (((mesh.nodes(:,1,:)+cylinder.x).^2 + (mesh.nodes(:,2,:)+cylinder.y).^2) < cylinder.r^2) & (cylinder.height/2 < mesh.nodes(:,3,:)) & (mesh.nodes(:,3,:) < cylinder.height/2);



if isfield(cylinder, 'muax') && isfield(cylinder, 'musx')
    kappax = 1./(3*(cylinder.muax+cylinder.musx));
    mesh.kappax(indices) = kappax;
end
if isfield(cylinder, 'muam') && isfield(cylinder, 'musm')
    kappam = 1./(3*(cylinder.muam+cylinder.musm));
    mesh.kappam(indices) = kappam;
end
if isfield(cylinder, 'muax')
    mesh.muax(indices) = cylinder.muax;
end
if isfield(cylinder, 'musx')
    mesh.musx(indices) = cylinder.musx;
end
if isfield(cylinder, 'ri')
    mesh.ri(indices) = cylinder.ri;
     mesh.c(indices)=(3e11/cylinder.ri);
end
if isfield(cylinder, 'muam')
    mesh.muam(indices) = cylinder.muam;
end
if isfield(cylinder, 'musm')
    mesh.musm(indices) = cylinder.musm;
end
if isfield(cylinder, 'muaf')
    % update excitation absorption if it isn't specified
    if ~isfield(cylinder, 'muax')
        old_muaf = mesh.muaf(indices);
        old_muax = mesh.muax(indices);
        mesh.muax(indices) = old_muax(1) + (cylinder.muaf - old_muaf(1));
    end
    
    mesh.muaf(indices) = cylinder.muaf;
end
if isfield(cylinder, 'tau')
    mesh.tau(indices) = cylinder.tau;
end
if isfield(cylinder, 'eta')
    mesh.eta(indices) = cylinder.eta;
end
if (isfield(cylinder, 'region') ~= 0)
    mesh.region(indices) = cylinder.region;
end
disp(['Number of nodes modified = ' ...
  num2str(length(indices))]);

