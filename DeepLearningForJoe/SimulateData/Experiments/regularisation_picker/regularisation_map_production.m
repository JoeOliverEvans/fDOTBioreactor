addpath(genpath('/home/jxe094/NIRFASTer'))
addpath(genpath('../../JoeScipts'))
addpath(genpath('../..'))
addpath(genpath('/home/jxe094/Work/DeepLearningForJoe/OptionalPatches'))
addpath(genpath('../../../../../NIRFASTer'))
addpath(genpath('../../../../NIRFASTer_FLpatch3'))
clear

mesh = load_mesh('cylinder/cylinder_dense');


mesh.muax = 5e-4*ones(size(mesh.muax));
mesh.muam = mesh.muax;
mesh.musx = 0.1*ones(size(mesh.musx));
mesh.musm = mesh.musx;
mesh.kappax = 1./(3*(mesh.muax + mesh.musx));
mesh.kappam = 1./(3*(mesh.muam + mesh.musm));

%%
samples = 150;
mesh.muaf = ones(size(mesh.muaf)) * 1e-10; % background fluorescence
num_nodes = size(mesh.nodes, 1);
max_blobs = 10;
blob_r_rng = [7,15];    % mm
% blob_muaf_rng = [3,7];   % times baseline
blob_muaf_rng = [1e-3,1e-1];   % mm-1; eta=0.4 in this mesh
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
opt.GPU = -2;

xgrid = linspace(-65,65,48);
ygrid = linspace(-65,65,48);
zgrid = linspace(-75,75,56);
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
noisy_img = zeros(48,48,56,samples);

[J, data0]= jacobiangrid_fl(mesh,[],[],[],0,solver, opt);
J = rmfield(J, 'completem');


globalsimvals = [];
regularisations = [];

for rep = 1:samples
    fprintf('%d/%d\n', rep, samples);
    mesh2 = mesh;
    num_blob = fix((rep-1)/10) + 1; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    blob_r = rand(num_blob,1) * (blob_r_rng(2) - blob_r_rng(1)) + blob_r_rng(1);
    blob_muaf = rand(num_blob,1) * (blob_muaf_rng(2) - blob_muaf_rng(1)) + blob_muaf_rng(1);
    blob_x = nan;
    blob_y = nan;
    blob_z = nan;
    while 1
        idx = true(num_blob, num_blob);
        tmp = pdist2([blob_x, blob_y, blob_z], [blob_x, blob_y, blob_z]) > blob_r+blob_r'+2;
        if all(blob_x.^2+blob_y.^2 < (blob_r-radius).^2) && (num_blob ==1 || all(tmp(tril(idx,-1))))
            break
        end
        blob_x = rand(num_blob,1) * (boundary(2)-boundary(1)) + boundary(1);
        blob_y = rand(num_blob,1) * (boundary(4)-boundary(3)) + boundary(3);
        blob_z = rand(num_blob,1) .* (boundary(6)-boundary(5)-blob_r-2) + (boundary(5)+blob_r);
    end
    
    fluctuate = 0.1*(rand(2,1)-0.5); % fluctuate the background mua by +/-10%
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
        mesh2 = add_blob(mesh2, blob);
    end
    mesh2.kappax = 1./(3*(mesh2.muax + mesh2.musx));
    mesh2.kappam = 1./(3*(mesh2.muam + mesh2.musm));

    try
        data = femdata_fl(mesh2, 0, solver,opt);
    catch
        data = femdata_fl(mesh2, 0, solver,opt);
    end
    all_muaf(:,rep) = mesh2.muaf;
    noise_lv = rand(2,1)*0.01;
    all_noise(:,rep) = noise_lv;
    all_datafl_clean(:,rep) = data.amplitudefl;
    all_datax_clean(:,rep) = data.amplitudex;
    all_datafl(:,rep) = data.amplitudefl + noise_lv(1)*max(data.amplitudefl).*randn(size(data.amplitudefl));
    all_datax(:,rep) = data.amplitudex + noise_lv(2)*max(data.amplitudex).*randn(size(data.amplitudex));
    all_nblob(rep) = num_blob;
    all_x(1:num_blob,rep) = blob_x;
    all_y(1:num_blob,rep) = blob_y;
    all_z(1:num_blob,rep) = blob_z;
    all_fluctuate(:,rep) = fluctuate;
    
    
   
    threshold_fl = 0.1*max(data.amplitudefl);
    threshold_xx = 0.1*max(data.amplitudex);
    idx = ~(data.amplitudefl<threshold_fl | data.amplitudex<threshold_xx);
    
    reg_strength_exponent_guess = (2.76095494 * sum(data.amplitudefl + noise_lv(1)*max(data.amplitudefl).*randn(size(data.amplitudefl))) + -56.51690199 * sum(data.amplitudex + noise_lv(2)*max(data.amplitudex).*randn(size(data.amplitudex))) + 30.45021654);
    
    upper_reg_exponent = ceil(reg_strength_exponent_guess);
    lower_reg_exponent = floor(reg_strength_exponent_guess);

    recon_grid_upper = tikhonov(J.complexm(idx,:)./data0.amplitudex(idx), 10^(upper_reg_exponent), data.amplitudefl(idx)./data.amplitudex(idx));
    recon_grid_lower = tikhonov(J.complexm(idx,:)./data0.amplitudex(idx), 10^(lower_reg_exponent), data.amplitudefl(idx)./data.amplitudex(idx));
    
    recon_grid_upper = reshape(recon_grid_upper, length(xgrid), length(ygrid), length(zgrid));
    recon_grid_lower = reshape(recon_grid_lower, length(xgrid), length(ygrid), length(zgrid));
    
    ref = reshape(mesh.vol.mesh2grid*mesh2.muaf, 48,48,56);

    [ssimval_upper,ssimmap_upper] = ssim(recon_grid_upper,ref);
    [ssimval_lower,ssimmap_lower] = ssim(recon_grid_lower,ref);
    
    ssimvals = [ssimval_upper, ssimval_lower];

    ongoing = true;

    reg_exponent = [10^ceil(reg_strength_exponent_guess)];
    reg_exponents = [reg_exponent, 10^lower_reg_exponent];

    while ongoing
         [value, index] = max(ssimvals);
         if ismember(reg_exponents(index)/10, reg_exponents) == false
            reg_exponents = [reg_exponents, reg_exponents(index)/10];
            recon_grid_new = tikhonov(J.complexm(idx,:)./data0.amplitudex(idx), reg_exponents(index)/10, data.amplitudefl(idx)./data.amplitudex(idx));
            recon_grid_new = reshape(recon_grid_new, length(xgrid), length(ygrid), length(zgrid));
            [ssimval_new,ssimmap_new] = ssim(recon_grid_new,ref);
            ssimvals = [ssimvals, ssimval_new];

         elseif ismember(reg_exponents(index)*10, reg_exponents) == false
            reg_exponents = [reg_exponents, reg_exponents(index)*10];
            recon_grid_new = tikhonov(J.complexm(idx,:)./data0.amplitudex(idx), reg_exponents(index)*10, data.amplitudefl(idx)./data.amplitudex(idx));
            recon_grid_new = reshape(recon_grid_new, length(xgrid), length(ygrid), length(zgrid));
            [ssimval_new,ssimmap_new] = ssim(recon_grid_new,ref);
            ssimvals = [ssimvals, ssimval_new];
         else
            ongoing = false;
            globalsimvals = [globalsimvals, max(ssimvals)];
            regularisations = [regularisations, max(reg_exponents)];
         end
    end
end



save('regularisation_information','all_datafl', 'all_datax', 'globalsimvals', 'regularisations','-v7.3')
clear
