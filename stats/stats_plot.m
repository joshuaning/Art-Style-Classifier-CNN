clear;
clc;
clf;

stats = csvread("stats.csv");

running_acc = stats(:,3);
epoch1 = running_acc(1:1563);
epoch2 = running_acc(1565:3127);
epoch3 = running_acc(3128:end);

epoch1_running_acc_new = conv(epoch1, [1,-1]);
epoch2_running_acc_new = conv(epoch2, [1,-1]);
epoch3_running_acc_new = conv(epoch3, [1,-1]);

epoch1_running_acc_new = epoch1_running_acc_new(1:end-1)/32;
epoch2_running_acc_new = epoch2_running_acc_new(1:end-1)/32;
epoch3_running_acc_new = epoch3_running_acc_new(1:end-1)/32;

filter = ones(1,100)/100;
e_tot = cat(1,epoch1_running_acc_new,epoch2_running_acc_new,epoch3_running_acc_new);
e_tot_f = conv(e_tot, filter, 'valid');

% plot(stats(2:end,1), e_tot, 'linewidth', 2); 
scatter(stats(2:end,1), e_tot, 3); 

hold on;
plot(100:length(e_tot_f)+99, e_tot_f, 'linewidth', 2); 
xlabel('Number of Batches(n)');
ylabel('Batch Accuracy');
title('Batch Accuracy of mini VGG-BN in 3 Epochs');
legend('batch accuracy', 'accuracy moving average(n=100)')


