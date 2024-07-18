load('bnd')

mesh = load_mesh('cylinder_large');
% mesh = load_mesh('cylinder_dense');

nodes=mesh.nodes;
mesh2=mesh;
mesh2.muaf=0.001*exp(-sum((nodes-[40,0,0]).^2,2)/100);
data=femdata_fl(mesh2,0);

xgrid = linspace(-65,65,48);
ygrid = linspace(-65,65,48);
zgrid = linspace(-75,75,56);
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
[J, data0]= jacobiangrid_fl(mesh);
J = rmfield(J, 'completem');

idx=data.amplitudefl>1e-10 & data.amplitudex>1e-9;
recon1=tikhonov(J.complexm(idx,:)./data0.amplitudex(idx), 1, data.amplitudefl(idx)./data.amplitudex(idx));
recon1=reshape(recon1,48,48,56);

volumeViewer(recon1/std(recon1(:))+bnd)
