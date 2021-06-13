pc = pcread('../data/test_cloud_occluded.pcd');
X0 = readmatrix('../data/Xinit.csv');
Xspr = readmatrix('../data/Xspr.csv');
Xsprb = readmatrix('../data/Xsprb.csv');
figure('Name', 'Registration result of SPR')
plot3(pc.Location(:,1), pc.Location(:,2), pc.Location(:,3), '.')
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal

hold on
plot3(X0(:,1), X0(:,2), X0(:,3), 'o')
plot3(Xspr(:,1), Xspr(:,2), Xspr(:,3), 'x')
legend('point cloud', 'initial', 'registered SPR')

figure('Name', 'Registration result of modified SPR')
plot3(pc.Location(:,1), pc.Location(:,2), pc.Location(:,3), '.')
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal

hold on
plot3(X0(:,1), X0(:,2), X0(:,3), 'o')
plot3(Xsprb(:,1), Xsprb(:,2), Xsprb(:,3), 'x')
legend('point cloud', 'initial', 'registered mod. SPR')