function [ out ] = HomogenousProduct(n, m)

out = nchoosek(n+m-1, m);