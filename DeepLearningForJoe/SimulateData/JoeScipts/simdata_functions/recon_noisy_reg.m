function [noisy_img, data0] = recon_noisy_reg(mesh, solver, opt, all_datafl_clean, all_datax_clean, samples, reg_strength)
%RECON_NOISY_REG Summary of this function goes here
%   Detailed explanation goes here
[J, data0]= jacobiangrid_fl(mesh,[],[],[],0,solver, opt);
J = rmfield(J, 'completem');
[~, invop] = tikhonov(J.complexm./data0.amplitudex,reg_strength);
tmp = all_datafl_clean./all_datax_clean;
maxamp = max(tmp);
norm_noise = 0.02*rand(samples, 1);
noisestd = maxamp' .* norm_noise;
noise = abs(randn(size(tmp,1), samples)*diag(noisestd));
% noise(noise<0) = 1e-20;

recon = invop*(tmp);
noisy_img = reshape(recon,48,48,56,samples);
