function [mesh, num_nodes, boundary, radius, all_muaf, all_datax, all_datafl, all_datax_clean, all_datafl_clean, all_x, all_y, all_z, all_nblob, all_noise, all_fluctuate, solver, opt, xgrid, ygrid, zgrid, noisy_img] = initial_setup(mesh, samples, max_blobs)
    mesh.muaf = ones(size(mesh.muaf)) * 1e-10; % background fluorescence
    num_nodes = size(mesh.nodes, 1);
    % blob_muaf_rng = [3,7];   % times baseline
    boundary = [min(mesh.nodes(:,1)), max(mesh.nodes(:,1)), min(mesh.nodes(:,2)), max(mesh.nodes(:,2)), min(mesh.nodes(:,3)), max(mesh.nodes(:,3))];
    radius = boundary(2);
    
    all_muaf = zeros(size(mesh.nodes,1), samples);
    all_datax = zeros(size(mesh.link,1), samples);
    all_datafl = zeros(size(mesh.link,1), samples);
    all_datax_clean = zeros(size(mesh.link,1), samples);
    all_datafl_clean = zeros(size(mesh.link,1), samples);
    
    all_x = zeros(max_blobs, samples);
    all_y = zeros(max_blobs, samples);
    all_z = zeros(max_blobs, samples);
    all_nblob = zeros(samples,1);
    all_noise = zeros(2, samples);
    all_fluctuate = zeros(2, samples);
    
    solver=get_solver('BiCGStab_GPU');
    opt = solver_options;
    opt.GPU = -1;
    
    xgrid = linspace(-65,65,48);
    ygrid = linspace(-65,65,48);
    zgrid = linspace(-75,75,56);
    mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
    noisy_img = zeros(48,48,56,samples);
end