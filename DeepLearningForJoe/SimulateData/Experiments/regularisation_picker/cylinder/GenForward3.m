addpath(genpath('../../NIRFASTer'))
clear

sizevar=[];
sizevar.r=65;
sizevar.height=150;
sizevar.xc=0;
sizevar.yc=0;
sizevar.zc=0;
sizevar.dist=1.5;

mesh = make_cylinder(sizevar);
nnode = size(mesh.nodes,1);
mesh.type = 'fluor';
mesh.muax = ones(nnode,1) * 0.0089;    % mm-1
mesh.musx = ones(nnode,1) * 1.314;
mesh.ri = ones(nnode,1) * 1.33;
mesh.muam = ones(nnode,1) * 0.0062;
mesh.musm = ones(nnode,1) * 1.274;
mesh.muaf = ones(nnode,1) * 0.002;
mesh.eta = ones(nnode,1) * 0.4;
mesh.tau = zeros(nnode,1);

radius = sizevar.r;
for i = 0:15
    srcx(i+1) = radius * cos(2*pi*i/16);
    srcy(i+1) = radius * sin(2*pi*i/16);
end

for i = 1:2:31
    detx((i+1)/2) = radius * cos(2*pi*i/32);
    dety((i+1)/2) = radius * sin(2*pi*i/32);
end

tmp = ones(16, 1) * (-70:20:70);
src = [repmat([srcx', srcy'], 8, 1), tmp(:)];

tmp = ones(16, 1) * (-60:20:60);
det = [repmat([detx', dety'], 7, 1), tmp(:)];
link = [];
for i=1:length(src)
    for j=1:length(det)
        if norm(src(i,:)-det(j,:))>5 && norm(src(i,:)-det(j,:))<90 % set maximum channel length. Adjust when necessary
            link = [link; [i, j, 1]];
        end
    end
end

mesh.source.fixed = 0;
mesh.source.num = (1:size(src,1))';
mesh.source.fwhm = zeros(size(src,1),1);
mesh.source.coord = src;

mesh.meas.fixed = 0;
mesh.meas.num = (1:size(det,1))';
mesh.meas.fwhm = zeros(size(det,1),1);
mesh.meas.coord = det;

mesh.link = link;

save_mesh(mesh, 'cylinder_dense')
