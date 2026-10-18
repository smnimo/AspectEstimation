classdef prob < handle
  properties
    vz
    zs
    L
  end
  methods
    function obj = prob()
    end
    function [] = reset(varargin)
      obj.vz = [];
      obj.zs = [];
      obj.L = [];
    end
    function [] = calcL(varargin)
      obj = varargin{1};
      session = varargin{2};
      S = length(session);
      Ls = zeros(S, 1);
      for s = 1 : S
        prv = session(s).prv;
%        N = size(prv, 2);
        f = session(s).f;
        T = length(f);
        for t = 1 : T
%            Ls(s) = Ls(s) + log(obj.zs(s, :) * obj.vz * [prv(:, f(t)); session(s).col(t, :)']);
            Ls(s) = Ls(s) + log(obj.zs(s, :) * obj.vz * prv(:, f(t)));
        end
      end
      obj.L = mean(Ls);
    end
  end
end
