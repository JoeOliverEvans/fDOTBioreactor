clear
%load('test_processed.mat')
% load('../SimData/3D/images3.mat')
% load('../SimulateData/Experiments/d_max10_vs_only10/test_processed_max10_testing.mat')
% load('../SimulateData/Experiments/d_max10_vs_only10/images3_blobs_testing.mat')
load('../SimulateData/Experiments/a_max3_vs_only3/images3_gaussian3_2500.mat')
% load('../SimulateData/Experiments/a_max3_vs_only3/test_processed_only3_testing.mat')
%load('Datasets/Gaussian/Gaussian_3/test_processed.mat')
%load('Datasets/Gaussian/Gaussian_3/images3_gaussian3_2500.mat')
load('bnd.mat')

recon1 = noisy_img(:,:,:,1:end);
truth = clean_img(:,:,:,1:end);
clear noisy_img clean_img

cmap = hot(128); % colormap

thresh = 0; % Since our data is normalized, this effectively says we only care about voxels above one standard deviation

%%%% Change this to something you need
i = 1;
tmp=recon1(:,:,:,1);
tmp2=recon1(:,:,:,2);
tmp3=recon1(:,:,:,3);
tmp4=recon1(:,:,:,4);
tmp5=truth(:,:,:,1);
%%%%

tmp(tmp<thresh) = 0;
% bnd is the boundary of the cylinder. Added only for nicer visualization
vol1 = volshow(tmp+bnd, Colormap=cmap); %tmp+bnd
cfg1 = vol1.Parent;
cfg1.CameraPosition=[0,-120,80];
cfg1.CameraUpVector=[0,0,1];
cfg1.BackgroundColor = [1,1,1];
cfg1.BackgroundGradient = 0;


tmp2(tmp2<thresh) = 0;
% bnd is the boundary of the cylinder. Added only for nicer visualization
vol2 = volshow(tmp2+bnd, Colormap=cmap); %tmp+bnd
cfg2 = vol2.Parent;
cfg2.CameraPosition=[0,-120,80];
cfg2.CameraUpVector=[0,0,1];
cfg2.BackgroundColor = [1,1,1];
cfg2.BackgroundGradient = 0;

tmp3(tmp3<thresh) = 0;
% bnd is the boundary of the cylinder. Added only for nicer visualization
vol2 = volshow(tmp3+bnd, Colormap=cmap); %tmp+bnd
cfg2 = vol2.Parent;
cfg2.CameraPosition=[0,-120,80];
cfg2.CameraUpVector=[0,0,1];
cfg2.BackgroundColor = [1,1,1];
cfg2.BackgroundGradient = 0;

tmp4(tmp4<thresh) = 0;
% bnd is the boundary of the cylinder. Added only for nicer visualization
vol2 = volshow(tmp4+bnd, Colormap=cmap); %tmp+bnd
cfg2 = vol2.Parent;
cfg2.CameraPosition=[0,-120,80];
cfg2.CameraUpVector=[0,0,1];
cfg2.BackgroundColor = [1,1,1];
cfg2.BackgroundGradient = 0;

tmp5(tmp5<thresh) = 0;
% bnd is the boundary of the cylinder. Added only for nicer visualization
vol2 = volshow(tmp5+bnd, Colormap=cmap); %tmp+bnd
cfg2 = vol2.Parent;
cfg2.CameraPosition=[0,-120,80];
cfg2.CameraUpVector=[0,0,1];
cfg2.BackgroundColor = [1,1,1];
cfg2.BackgroundGradient = 0;