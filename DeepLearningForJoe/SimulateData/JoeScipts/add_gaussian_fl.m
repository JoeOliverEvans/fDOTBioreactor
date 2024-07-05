function mesh = add_gaussian_fl(mesh, gaussian)

% mesh = add_blob_fl(mesh, gaussian)
%
% Adds circular (2D) and spherical (3D) fluorescence 
% anomalies to the mesh.
%
% mesh is the mesh variable or filename.
% gaussian contains the anomaly info.
% gaussian should have the following format:
%
% gaussian.x - x position
% gaussian.y - y position
% gaussian.z - z position (optional)
% gaussian.sigma - radius
% gaussian.muax - absorption coefficient at excitation
% gaussian.musx - scatter coefficient at excitation
% gaussian.muam - absorption coefficient at emission
% gaussian.musm - scatter coefficient at emission
% gaussian.muaf - absorption of fluorophore
% gaussian.eta - quantum yield of fluorophore
% gaussian.tau - lifetime of fluorophore
% gaussian.ri - refractive index
% gaussian.region - region number
% gaussian.dist - distance between nodes for anomaly (BEM only)



% If not a workspace variable, load mesh
if ischar(mesh)== 1
  mesh = load_mesh(mesh);
end


if (isfield(gaussian, 'x') == 0)
    errordlg('No x coordinate was given for the anomaly','NIRFAST Error');
    error('No x coordinate was given for the anomaly');
end
if (isfield(gaussian, 'y') == 0)
    errordlg('No y coordinate was given for the anomaly','NIRFAST Error');
    error('No y coordinate was given for the anomaly');
end
if (isfield(gaussian, 'z') == 0 || mesh.dimension == 2)
    gaussian.z = 0;
end
if (isfield(gaussian, 'sigma') == 0)
    errordlg('No radius was given for the anomaly','NIRFAST Error');
    error('No radius was given for the anomaly');
end

% dist = distance(mesh.nodes(:,1:3),ones(length(mesh.bndvtx),1),[gaussian.x gaussian.y gaussian.z]);
dist = pdist2(mesh.nodes, [gaussian.x, gaussian.y, gaussian.z]);
gaussian_coefficient = exp(-(dist).^2./(2*gaussian.sigma.^2));
gaussian_coefficient(isnan(gaussian_coefficient)) = 0;

if isfield(gaussian, 'muax') && isfield(gaussian, 'musx')
    kappax = 1./(3*(gaussian.muax+gaussian.musx));
    mesh.kappax = mesh.kappax + kappax * gaussian_coefficient;
end
if isfield(gaussian, 'muam') && isfield(gaussian, 'musm')
    kappam = 1./(3*(gaussian.muam+gaussian.musm));
    mesh.kappam = mesh.kappam + kappam * gaussian_coefficient;
end
if isfield(gaussian, 'muax')
    mesh.muax = mesh.muax + gaussian.muax* gaussian_coefficient;
end
if isfield(gaussian, 'musx')
    mesh.musx = mesh.musx + gaussian.musx * gaussian_coefficient;
end
if isfield(gaussian, 'ri')
    mesh.ri = mesh.ri + gaussian.ri * gaussian_coefficient;
     mesh.c = mesh.c + (3e11/gaussian.ri) * gaussian_coefficient;
end
if isfield(gaussian, 'muam')
    mesh.muam = mesh.muam + gaussian.muam * gaussian_coefficient;
end
if isfield(gaussian, 'musm')
    mesh.musm = mesh.musm + gaussian.musm * gaussian_coefficient;
end
if isfield(gaussian, 'muaf')
    % update excitation absorption if it isn't specified
    if ~isfield(gaussian, 'muax')
        mesh.muax = mesh.muax + gaussian.muaf * gaussian_coefficient;
    end
    
    mesh.muaf = mesh.muaf + gaussian.muaf * gaussian_coefficient;
end
if isfield(gaussian, 'tau')
    mesh.tau = mesh.tau + gaussian.tau * gaussian_coefficient;
end
if isfield(gaussian, 'eta')
    mesh.eta = mesh.eta + gaussian.eta * gaussian_coefficient;
end


