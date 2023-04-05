clear;
clc;
clf;

stats = csvread("stats.csv");
plot(stats(:,1), stats(:,3)./stats(:,4)); 