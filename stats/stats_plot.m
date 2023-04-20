%% clear stuff
clear;
clc;
clf;
close all;

% mini vgg ----------------------------------------------------------------
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


% VGG-16 BN batch size 16 -------------------------------------------------

window = 100;

stats1 = csvread("stats_saves_vgg_16/stats_epoch_16.csv");
stats2 = csvread("stats_saves_vgg_16/stats_epoch_50.csv");
stats = [stats1(:,2); stats2(:,2)]/16;
test_stats = csvread("test_eval_stats.csv");

filter = ones(1,window)/window;
stats_f = conv(stats, filter, 'valid');

figure();
scatter(1:length(stats), stats, 3); 

hold on;
plot(window:length(stats_f)+window-1, stats_f, 'linewidth', 2); 
plot(test_stats(:,1)*3125, test_stats(:,2)/100, 'linewidth', 3);

xlabel('Number of Batches(n)');
ylabel('Batch Accuracy');
title('Accuracy of VGG16-BN in 50 Epochs Trained with Batch Size 16');
legend('batch accuracy', 'batch accuracy moving average(n=100)', ...
        'test accuracy')

figure();
plot(100:length(e_tot_f)+99, e_tot_f, 'linewidth', 2); 
hold on
plot((window:length(stats_f)+window-1)/2, stats_f, 'linewidth', 2); 
axis([0 3*1563 0 0.5])
xline(1563:1563:3*1563,'-',{'epoch'})
title('mini VGG vs VGG-16 (Batch Size = 16)');
legend('mini VGG batch accuracy moving avg', 'VGG 16 batch accuracy moving avg')
xlabel('Number of Batches Normalized(n)');
ylabel('Batch Accuracy');


% VGG-16 BN batch size 32 -------------------------------------------------
stats3 = csvread("stats_saves_vgg16_32/stats_epoch_17.csv");
stats4 = csvread("stats_saves_vgg16_32/stats_epoch_30.csv");
stats_32 = [stats3(:,2); stats4(:,2)]/32;
test_stats_32 = csvread("test_eval_stats_32.csv");

filter = ones(1,window)/window;
stats_f_32 = conv(stats_32, filter, 'valid');

figure();
plot((window:length(stats_f)+window-1)/2, stats_f, 'linewidth', 2);
hold on;
plot(window:length(stats_f_32)+window-1, stats_f_32, 'linewidth', 2);
plot(test_stats(:,1)*3125 / 2, test_stats(:,2)/100, 'linewidth', 3);
plot(test_stats_32(:,1)*1563, test_stats_32(:,2)/100, 'linewidth', 3);
xline(1563:1563:30*1563,'-',{'epoch'})
axis([0 30*1563 0 1])


xlabel('Number of Batches Normalized(n)');
ylabel('Batch Accuracy');
title('Accuracy of VGG16-BN Batch Size 16 vs 32');
legend('Batch Accuracy Moving Avg Batch Size 16', ...
        'Batch Accuracy Moving Avg Batch Size 32', ...
        'Test Accuracy Batch Size 16', ...
        'Test Accuracy Batch Size 32')
    
figure();
scatter(1:length(stats_32), stats_32, 3); 
hold on;
plot(window:length(stats_f_32)+window-1, stats_f_32, 'linewidth', 2); 
plot(test_stats_32(:,1)*1563, test_stats_32(:,2)/100, 'linewidth', 3);

xlabel('Number of Batches(n)');
ylabel('Batch Accuracy');
title('Accuracy of VGG16-BN in 30 Epochs Trained with Batch Size 32');
legend('batch accuracy', 'batch accuracy moving average(n=100)', ...
        'test accuracy')

