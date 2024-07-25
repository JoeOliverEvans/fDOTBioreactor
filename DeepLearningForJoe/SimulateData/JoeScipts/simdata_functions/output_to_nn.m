function output_to_nn(name, mesh, all_muaf, samples, all_datafl_clean, all_datax_clean, noisy_img, all_x, all_y, all_z, all_nblob, all_datafl, all_datax, all_noise, all_fluctuate)
%OUTPUT_TO_NN Summary of this function goes here
%   Detailed explanation goes here
all_muaf2 = mesh.vol.mesh2grid*all_muaf;
clean_img = reshape(all_muaf2,48,48,56,samples);

tmp = all_datafl_clean./all_datax_clean;
maxamp = max(tmp);
norm_noise = 0.02*rand(samples, 1);
noisestd = maxamp' .* norm_noise;
noise = abs(randn(size(tmp,1), samples)*diag(noisestd));

inmesh = mesh.vol.gridinmesh;

for i=1:samples
    tmp=clean_img(:,:,:,i);
    clean_img(:,:,:,i)=tmp/std(tmp(:));
end
for i=1:samples
    tmp=noisy_img(:,:,:,i);
    noisy_img(:,:,:,i)=tmp/std(tmp(:));
end
mask=zeros(48,48,56);
mask(inmesh)=1;

save(name, 'clean_img', 'noisy_img', 'inmesh','all_x', 'all_y', 'all_z', 'all_nblob', 'all_muaf', 'all_datafl', 'all_datax', 'all_noise', 'all_fluctuate', 'all_datax_clean', 'all_datafl_clean','norm_noise','noise','mask', '-v7.3')
clear
