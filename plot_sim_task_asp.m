clear
close all

load('results/cor.mat')
bar([diag(cor_all{3}), diag(cor{3})]')