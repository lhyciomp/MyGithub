function  fgSal= calfgDistributionSal(salSup,Isum,x_vals,y_vals,m,n,spnum,number)

comSal = calDistribution(salSup,Isum,x_vals,y_vals,m,n,spnum,number);
fgSal = comSal;


index =comSal > mean(comSal);


%mean是对comSal的每个元素进行求和，comSal是个cenNum*1的列向量，
%大于号的返回值是一个cenNum*1的行向量,大于时，对应位置赋值为1（背景点），否则为0（前景点）

fgSal(index) = mean(comSal);%背景点赋值为平均值；前景点不变

fgSal = 1 - normalize(fgSal);%comSal大于平均值的背景点，全部被赋值为0；而小于平均值的前景点保持不变

clear dist coherence centric
