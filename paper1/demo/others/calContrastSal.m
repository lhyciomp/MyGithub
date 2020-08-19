function Thr_ConSal = calContrastSal( ConSal)
%CALCONTRASTSAL 此处显示有关此函数的摘要
%   此处显示详细说明
  Thr_ConSal=ConSal;
  index=ConSal<mean(ConSal);
  Thr_ConSal(index)=0;
end

