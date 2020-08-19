function  d = mydist(X,Y)  
% 计算向量Y到向量X的欧氏距离的开方  
d = sqrt(sum((X-Y).^2));  
end  