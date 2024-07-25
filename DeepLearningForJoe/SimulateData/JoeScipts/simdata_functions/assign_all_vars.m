function [all_muaf, all_noise, all_datafl_clean, all_datax_clean, all_datafl, all_datax, all_nblob, all_x, all_y, all_z, all_fluctuate] = assign_all_vars(mesh2, data, num_blob, blob_x, blob_y, blob_z, fluctuate, rep, all_muaf, all_noise, all_datafl_clean, all_datax_clean, all_datafl, all_datax, all_nblob, all_x, all_y, all_z, all_fluctuate)
%ASSIGN_ALL_VARS Summary of this function goes here
%   Detailed explanation goes here
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
