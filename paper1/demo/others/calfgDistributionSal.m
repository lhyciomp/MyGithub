function  fgSal= calfgDistributionSal(salSup,Isum,x_vals,y_vals,m,n,spnum,number)

comSal = calDistribution(salSup,Isum,x_vals,y_vals,m,n,spnum,number);
fgSal = comSal;


index =comSal > mean(comSal);


%mean�Ƕ�comSal��ÿ��Ԫ�ؽ�����ͣ�comSal�Ǹ�cenNum*1����������
%���ںŵķ���ֵ��һ��cenNum*1��������,����ʱ����Ӧλ�ø�ֵΪ1�������㣩������Ϊ0��ǰ���㣩

fgSal(index) = mean(comSal);%�����㸳ֵΪƽ��ֵ��ǰ���㲻��

fgSal = 1 - normalize(fgSal);%comSal����ƽ��ֵ�ı����㣬ȫ������ֵΪ0����С��ƽ��ֵ��ǰ���㱣�ֲ���

clear dist coherence centric
