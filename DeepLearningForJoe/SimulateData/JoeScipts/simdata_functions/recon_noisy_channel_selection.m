function [noisy_img, data0, J] = recon_noisy_channel_selection(rep, mesh, solver ,opt, data, xgrid, ygrid, zgrid, data0, noisy_img, reg_strength)
%RECON_NOISY_CHANNEL_SELECTION Summary of this function goes here
%   Detailed explanation goes here

%%%
threshold_fl = mean(data.amplitudefl) + 1*std(data.amplitudefl);
threshold_xx = mean(data.amplitudex) + 1*std(data.amplitudex);
%%%

[J, data0]= jacobiangrid_fl(mesh,[],[],[],0,solver, opt);
J = rmfield(J, 'completem');


tmp1 = data.amplitudefl./data.amplitudex;


idx = ~(data0.amplitudefl>threshold_fl & data0.amplitudex>threshold_xx);

tmp1 = tmp1(idx);

recon_grid = tikhonov(J.complexm(idx,:)./data0.amplitudex(idx), reg_strength, tmp1);

recon_grid = reshape(recon_grid, length(xgrid), length(ygrid), length(zgrid));

noisy_img(:,:,:,rep) = recon_grid;

