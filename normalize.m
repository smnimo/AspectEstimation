function A = normalize(N, dim)
% A = normalize(N, dim)
% N   : matrix
% dim : dim == 1|2, normalized column(1) or row(2)
% if normalize(N), all array sum to 1
%
% make normalize matrix.

if nargin < 2
    z = sum(N(:));
    s = z + (z==0);
    A = N / s;
elseif dim == 1
    z = sum(N, 1);
    s = z + (z==0);
    A = N ./ repmat(s, size(N, 1), 1);
elseif dim == 2
    z = sum(N, 2);
    s = z + (z==0);
    A = N ./ repmat(s, 1, size(N, 2));
else
  z=sum(N,dim);
  s = z + (z==0);
  L=size(N,dim);
  d=length(size(N));
  v=ones(d,1);
  v(dim)=L;
  c=repmat(s,v');
  A=N./c;
end

