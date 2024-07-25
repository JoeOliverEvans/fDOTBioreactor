function [mesh] = load_mesh_altered_properties(mesh_string)
%LOAD_MESH_ALTERED_PROPERTIES Summary of this function goes here
%   Detailed explanation goes here
mesh = load_mesh(mesh_string);

mesh.muax = 5e-4*ones(size(mesh.muax));
mesh.muam = mesh.muax;
mesh.musx = 0.1*ones(size(mesh.musx));
mesh.musm = mesh.musx;
mesh.kappax = 1./(3*(mesh.muax + mesh.musx));
mesh.kappam = 1./(3*(mesh.muam + mesh.musm));


