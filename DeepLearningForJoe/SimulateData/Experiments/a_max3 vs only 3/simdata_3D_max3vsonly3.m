addpath(genpath('/home/jxe094/NIRFASTer'))
addpath(genpath('../../JoeScipts'))
%addpath("../../JoeScipts\simdata_functions\")
addpath(genpath('/home/jxe094/Work/DeepLearningForJoe/OptionalPatches'))
addpath('../..')
clear

mesh = load_mesh_altered_properties('../../cylinder_large');

%% Max 3

%%%
samples=2500;
max_blobs=3;
blob_r_rng=[7,15];
blob_muaf_rng=[1e-3,1e-1];
channel_selection = true;
%%%

[mesh, num_nodes, boundary, radius, all_muaf, all_datax, all_datafl, all_datax_clean, all_datafl_clean, all_x, all_y, all_z, all_nblob, all_noise, all_fluctuate, solver, opt, xgrid, ygrid, zgrid, noisy_img] = initial_setup(mesh, samples,max_blobs);
data0 = [];
J = [];
for rep = 1:samples
    fprintf('%d/%d\n', rep, samples);
    mesh2 = mesh;

    %%%
    num_blob = randperm(max_blobs, 1);
    inclusion_type = 'blob';
    %%%

    blob_r = rand(num_blob,1) * (blob_r_rng(2) - blob_r_rng(1)) + blob_r_rng(1);
    blob_muaf = rand(num_blob,1) * (blob_muaf_rng(2) - blob_muaf_rng(1)) + blob_muaf_rng(1);
    fluctuate = 0.1*(rand(2,1)-0.5); % fluctuate the background mua by +/-10%

    [mesh2,blob_x, blob_y, blob_z] = add_inclusion(mesh, mesh2, inclusion_type, num_blob, boundary, blob_r, fluctuate, blob_muaf);

    try
        data = femdata_fl(mesh2, 0, solver,opt);
    catch
        data = femdata_fl(mesh2, 0, solver,opt);
    end

    [all_muaf, all_noise, all_datafl_clean, all_datax_clean, all_datafl, all_datax, all_nblob, all_x, all_y, all_z, all_fluctuate] = assign_all_vars(mesh2, data, num_blob, blob_x, blob_y, blob_z, fluctuate, rep, all_muaf, all_noise, all_datafl_clean, all_datax_clean, all_datafl, all_datax, all_nblob, all_x, all_y, all_z, all_fluctuate);

    if channel_selection == true
        [noisy_img, data0] = recon_noisy_channel_selection(rep, mesh, solver ,opt, data, xgrid, ygrid, zgrid, data0, noisy_img);
    end
end

if channel_selection == false
    [noisy_img, data0, J] = recon_noisy_reg(mesh, solver, opt, all_datafl_clean, all_datax_clean, samples, J);
end

name = 'images3_max3';

output_to_nn(name, mesh, all_muaf, samples, all_datafl_clean, all_datax_clean, noisy_img, all_x, all_y, all_z, all_nblob, all_datafl, all_datax, all_noise, all_fluctuate)


%% Only 3

[mesh, num_nodes, boundary, radius, all_muaf, all_datax, all_datafl, all_datax_clean, all_datafl_clean, all_x, all_y, all_z, all_nblob, all_noise, all_fluctuate, solver, opt, xgrid, ygrid, zgrid, noisy_img] = initial_setup(mesh, samples,max_blobs);
data0 = [];
J = [];
for rep = 1:samples
    fprintf('%d/%d\n', rep, samples);
    mesh2 = mesh;

    %%%
    num_blob = 3; %randperm(max_blobs, 1);
    inclusion_type = 'blob';
    %%%

    blob_r = rand(num_blob,1) * (blob_r_rng(2) - blob_r_rng(1)) + blob_r_rng(1);
    blob_muaf = rand(num_blob,1) * (blob_muaf_rng(2) - blob_muaf_rng(1)) + blob_muaf_rng(1);
    fluctuate = 0.1*(rand(2,1)-0.5); % fluctuate the background mua by +/-10%

    [mesh2,blob_x, blob_y, blob_z] = add_inclusion(mesh, mesh2, inclusion_type, num_blob, boundary, blob_r, fluctuate, blob_muaf);

    try
        data = femdata_fl(mesh2, 0, solver,opt);
    catch
        data = femdata_fl(mesh2, 0, solver,opt);
    end

    [all_muaf, all_noise, all_datafl_clean, all_datax_clean, all_datafl, all_datax, all_nblob, all_x, all_y, all_z, all_fluctuate] = assign_all_vars(mesh2, data, num_blob, blob_x, blob_y, blob_z, fluctuate, rep, all_muaf, all_noise, all_datafl_clean, all_datax_clean, all_datafl, all_datax, all_nblob, all_x, all_y, all_z, all_fluctuate);

    if channel_selection == true
        [noisy_img, data0] = recon_noisy_channel_selection(rep, mesh, solver ,opt, data, xgrid, ygrid, zgrid, data0, noisy_img);
    end
end

if channel_selection == false
    [noisy_img, data0, J] = recon_noisy_reg(mesh, solver, opt, all_datafl_clean, all_datax_clean, samples, J);
end

name = 'images3_only3';

output_to_nn(name, mesh, all_muaf, samples, all_datafl_clean, all_datax_clean, noisy_img, all_x, all_y, all_z, all_nblob, all_datafl, all_datax, all_noise, all_fluctuate)

