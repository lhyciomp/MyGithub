function [U, V,objFcn] = myfcm(data, c, T, m, epsm)  
% fuzzy c-means algorithm  
% 输入： data： 待聚类数据，n行s列，n为数据个数，s为每个数据的特征数  
%        c  ：  聚类中心个数  
%        m  :   模糊系数  
% 输出： U  :   隶属度矩阵，c行n列，元素uij表示第j个数据隶属于第i类的程度  
%        V  ：  聚类中心向量，c行s列，有c个中心，每个中心有s维特征  
% written by Zhang Jin  
% see also  :  mydist.m  myplot.m  
  
if nargin < 3  
    T = 100;   %默认迭代次数为100  
end  
if nargin < 5  
    epsm = 1.0e-6;  %默认收敛精度  
end  
if nargin < 4  
    m = 2;   %默认模糊系数值为2  
end  
  
[n, s] = size(data);   
% 初始化隶属度矩阵U(0),并归一化  
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
    % 更新V(t)  
    Um = U0.^m;  
    V = Um*data./(sum(Um,2)*ones(1,s));   % MATLAB矩阵相乘啊，好东西  
    % 更新U(t)  
    for i = 1:c  
        for j = 1:n  
            distance(i,j) = mydist(data(j,:),V(i,:));  
        end  
    end  
    U=1./(distance.^m.*(ones(c,1)*sum(distance.^(-m))));   
    objFcn(iter) = sum(sum(Um.*distance.^2));  
    % FCM算法停止条件  
    if norm(U-U0,Inf)<epsm    
        break  
    end    
    U0=U;  
end  