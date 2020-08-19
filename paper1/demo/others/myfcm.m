function [U, V,objFcn] = myfcm(data, c, T, m, epsm)  
% fuzzy c-means algorithm  
% ���룺 data�� ���������ݣ�n��s�У�nΪ���ݸ�����sΪÿ�����ݵ�������  
%        c  ��  �������ĸ���  
%        m  :   ģ��ϵ��  
% ����� U  :   �����Ⱦ���c��n�У�Ԫ��uij��ʾ��j�����������ڵ�i��ĳ̶�  
%        V  ��  ��������������c��s�У���c�����ģ�ÿ��������sά����  
% written by Zhang Jin  
% see also  :  mydist.m  myplot.m  
  
if nargin < 3  
    T = 100;   %Ĭ�ϵ�������Ϊ100  
end  
if nargin < 5  
    epsm = 1.0e-6;  %Ĭ����������  
end  
if nargin < 4  
    m = 2;   %Ĭ��ģ��ϵ��ֵΪ2  
end  
  
[n, s] = size(data);   
% ��ʼ�������Ⱦ���U(0),����һ��  
U0 = rand(c, n);  
temp = sum(U0,1);  
for i=1:n  
    U0(:,i) = U0(:,i)./temp(i);  
end  
iter = 0;   
V(c,s) = 0; U(c,n) = 0; distance(c,n) = 0;  
  
while( iter<T  )  
    iter = iter + 1;  
%    U =  U0;  
    % ����V(t)  
    Um = U0.^m;  
    V = Um*data./(sum(Um,2)*ones(1,s));   % MATLAB������˰����ö���  
    % ����U(t)  
    for i = 1:c  
        for j = 1:n  
            distance(i,j) = mydist(data(j,:),V(i,:));  
        end  
    end  
    U=1./(distance.^m.*(ones(c,1)*sum(distance.^(-m))));   
    objFcn(iter) = sum(sum(Um.*distance.^2));  
    % FCM�㷨ֹͣ����  
    if norm(U-U0,Inf)<epsm    
        break  
    end    
    U0=U;  
end  