function [mesh2,blob_x, blob_y, blob_z] = add_inclusion(mesh, mesh2, inclusion_type, num_blob, boundary, blob_r, fluctuate, blob_muaf)
%ADD_INCLUSION Summary of this function goes here
%   Detailed explanation goes here
blob_x = rand(num_blob,1) * (boundary(2)-boundary(1)) + boundary(1);
blob_y = rand(num_blob,1) * (boundary(4)-boundary(3)) + boundary(3);
blob_z = rand(num_blob,1) .* (boundary(6)-boundary(5)-blob_r-2) + (boundary(5)+blob_r);

mesh2.muax = mesh.muax*(1+fluctuate(1));
mesh2.muam = mesh2.muax;
mesh2.musx = mesh.musx*(1+fluctuate(2));
mesh2.musm = mesh2.musx;

for i=1:num_blob
    blob=[];
    blob.x = blob_x(i);
    blob.y = blob_y(i);
    blob.z = blob_z(i);
    blob.r = blob_r(i);
    blob.muaf = blob_muaf(i);
    if strcmp(inclusion_type, 'blob')
        mesh2 = add_blob(mesh2, blob);
    elseif strcmp(inclusion_type, 'cylinder')
        %Cylinder position and rotation
        rotationAngles = [rand*360, rand*360, rand*360];
        translation = [-blob.x, -blob.y, -blob.z];
        blob.tform = rigidtform3d(rotationAngles, translation);

        blob.height = rand*75;
        mesh2 = add_cylinder_fl(mesh2, blob);
    elseif strcmp(inclusion_type, 'gaussian')
        blob.sigma = rand * 20;
        mesh2 = add_gaussian_fl(mesh2, blob);
    else
        error('inclusion function not specified')
    end
end
mesh2.kappax = 1./(3*(mesh2.muax + mesh2.musx));
mesh2.kappam = 1./(3*(mesh2.muam + mesh2.musm));

