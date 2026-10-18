classdef itemlist < handle
  properties
    num
    h
    w
    list
  end
  methods
    function obj = itemlist(varargin)
      if nargin == 2
        obj.h = varargin{1};
        obj.w = varargin{2};
        obj.num = obj.w * obj.h;
        obj.list = 1 : 15;
      elseif nargin == 3
        obj.h = varargin{1};
        obj.w = varargin{2};
        obj.num = obj.w * obj.h;
        obj.list = varargin{3};
      end
    end
    function [] = reset(varargin)
      obj = varargin{1};
      obj.num = [];
      obj.h = [];
      obj.w = [];
      obj.list = [];
    end
  end
end

