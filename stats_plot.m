clear;
clc;
clf;

stats = csvread("stats.csv");
plot(stats(:,1), stats(:,3)./stats(:,4), 'linewidth', 2); 

xlabel('Number of Batches(n)');
ylabel('Running Accuracy');
title('Running Accuracy of VGG-BN in 3 Epochs');