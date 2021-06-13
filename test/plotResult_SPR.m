pc = pcread('data/test_cloud_downsampled.pcd');
X0 = readmatrix('data/Xinit.csv')
Xfinal = readmatrix('data/Xregistered.csv')

figure('Name', 'Registration of test point cloud')
plot3(pc.Location(:,1), pc.Location(:,2), pc.Location(:,3), '.')
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal

hold all
plot3(X0(:,1), X0(:,2), X0(:,3), 'o')
plot3(Xfinal(:,1), Xfinal(:,2), Xfinal(:,3), 'o')
legend('measured', 'initial', 'registered C++')