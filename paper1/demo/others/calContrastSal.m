function Thr_ConSal = calContrastSal( ConSal)
%CALCONTRASTSAL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
  Thr_ConSal=ConSal;
  index=ConSal<mean(ConSal);
  Thr_ConSal(index)=0;
end

