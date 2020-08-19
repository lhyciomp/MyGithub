function demo4

clear ;
clc;
addpath('./others/');
%%------------------------set parameters---------------------%%
global imgRoot;
global saldir;
% global supdir;
global imnames;
% spnumber = 200;% superpixel number
spnumber=[120 160 200 250 300 ];
theta = 10; % control the edge weight
% cenNum = 36;
alpha = 0.99;
% options = zeros(1,5);
% options(1) = 1; % display
% options(2) = 1;
% options(3) = 0.00002; % precision
% options(4) = 200; % maximum number of iterations
imgRoot='./test/';% test image path
saldir='./saliencymap/';% the output path of the saliency map
       %%-------------------�������ɷ���ͼƬ���ļ���------------------%%
% supdir='./superpixels/';
% compdir='./salCompact/';
% condir='./salContrast/';
finaldir='./FinalSaliencyMap/';
% mkdir(supdir);
mkdir(saldir);
% mkdir(compdir);
mkdir(finaldir);
% mkdir(condir);
imnames=dir([imgRoot '*' '.jpg']);
%  Obj_Cen_Con=cell(1,5);
 sup_folderName=cell(1,5);
 com_folderName=cell(1,5);
 con_folderName=cell(1,5);
 sal_folderName=cell(1,5);
%  salall_Compact=cell(1,5);
%  salall_Contrast=cell(1,5);
%  InitSaliencyMap=cell(1,5);
%  Cen_folderName=cell(1,5);
%  InitCom_folderName=cell(1,5);


%   GX_con_folderName=cell(1,5);
guassSigmaRatio = 0.33; 
for q=1:length(spnumber)
     sup_folderName{q}=['./superpixels',num2str(q),'/'];

    com_folderName{q}=['./salCompact',num2str(q)];
%      ComCen_folderName{q}=['./ComCenMap',num2str(q)];
%     Gau_com_folderName{q}=['./salGauCompact',num2str(q)];
    con_folderName{q}=['./salContrast',num2str(q)];
%     ConCen_folderName{q}=['./ConCenMap',num2str(q)];
%      Gau_con_folderName{q}=['./salGauCon',num2str(q)];
    sal_folderName{q}=['./saliencymap',num2str(q)];
%     sal_Gauss_folderName{q}=['./saliencyGaussMap',num2str(q)];
%    SGX_sal_folderName{q}=['./GX_sal',num2str(q)];
%    InitCom_folderName{q}=['./ InitCom',num2str(q)];
%    InitCon_folderName{q}=['./ InitCon',num2str(q)];
%    InitCon_NoCom_folderName{q}=['./ InitConNoCom',num2str(q)];
%    sal_Cen_folderName{q}=['./ salCen',num2str(q)];
%     Cen_NoSal_folderName{q}=['./ NoSalCen',num2str(q)];
%     AllConSal_folderName{q}=['./ InitAllCon',num2str(q)];
    
    mkdir(sup_folderName{q});
    mkdir(com_folderName{q});
    mkdir(con_folderName{q});
    mkdir(sal_folderName{q});
%     mkdir( sal_Gauss_folderName{q});
%     mkdir(Cen_folderName{q});
%     mkdir(ComCen_folderName{q});
%     mkdir(ConCen_folderName{q});
%     mkdir( SGX_sal_folderName{q});
%   mkdir(Gau_com_folderName{q});
%   mkdir(Gau_con_folderName{q});
%    mkdir(InitCom_folderName{q});
%    mkdir(InitCon_folderName{q});
%     mkdir(InitCon_NoCom_folderName{q});
%     mkdir(sal_Cen_folderName{q});
%     mkdir(Cen_NoSal_folderName{q});
%      mkdir(AllConSal_folderName{q});
end

for ii=1:length(imnames)   
for M=1:length(spnumber)


    disp(ii);
    imname = [imgRoot imnames(ii).name]; 
    [input_im, w]=removeframe(imname);% run a pre-processing to remove the image frame 
    
    [m,n,k] = size(input_im);    
    input_vals=reshape(input_im, m*n, k);
    clear k

    %% ---------------------- generate superpixels Layer 1 ---------------------%% 
    imname=[imname(1:end-4) '.bmp'];
%     sup_outname=[sup_folderName{M},'/'];
    comm=['SLICSuperpixelSegmentation' ' ' imname ' ' int2str(20) ' ' int2str(spnumber(M)) ' ' sup_folderName{M}];%ʹ��SLIC�������inname�ָ������supdir�ļ�����
    system(comm);    %SLIC�ָ�
    spname=[sup_folderName{M} imnames(ii).name(1:end-4)  '.dat'];%�ָ���SLIC.dat�ļ�
    superpixels=ReadDAT([m,n],spname); % superpixel label matrix����ȡ�ָ���ͼƬ�г����ؿ����Ϣ�������Ӧ���Ǳ�ǩ��
    spnum=max(superpixels(:));% the actual superpixel number��ȡ�����صĸ����������������ǽ�ÿ�кϲ���һ��������������ǩ�ı�žʹ������صĸ���
    [adjc ,bord] = calAdjacentMatrix(superpixels,spnum); 

    clear comm spname
    
    
    % compute the feature (mean color in lab color space) for each superpixels    
    rgb_vals = zeros(spnum,3);%����һ��spum*3�������
    inds=cell(spnum,1);%����һ��spum*1�Ŀյ�Ԫ����
    [x, y] = meshgrid(1:1:n, 1:1:m);%��������㣨m*n��
    x_vals = zeros(spnum,1);%����һ��spum*1��ȫ�����
    y_vals = zeros(spnum,1);
    num_vals = zeros(spnum,1);
    for i=1:spnum
        inds{i}=find(superpixels==i);%�ҳ������i�������ؿ飬�������ؿ�ĵ��±�
        num_vals(i) = length(inds{i});%��ȡ�����ؿ��е�������Ŀ
        rgb_vals(i,:) = mean(input_vals(inds{i},:),1);%�Գ����ؿ�i�е�����RGBֵȡƽ����������ƽ����
        x_vals(i) = sum(x(inds{i}))/num_vals(i);%ȡ���������е�ÿ�����ص�xֵ������ƽ���������ؿ�i��������x��
        y_vals(i) = sum(y(inds{i}))/num_vals(i);%ȡ���������е�ÿ�����ص�yֵ������ƽ���������ؿ�i��������y��
    end  
    seg_vals = colorspace('Lab<-', rgb_vals);%��rgb��ɫ�ռ�ת��Ϊlab��ɫ�ռ�
    sp_vals=[x_vals y_vals];%spnum*2�ľ��������г����ؿ�Ŀռ���������
    clear i x y input_vals rgb_vals
    
    
    %�������ƶȾ���A��sim��
    disSupCen = normalize( DistanceZL(seg_vals, seg_vals, 'euclid'), 'column' );
    simSupCen = exp( -theta*disSupCen );%spnum*cenNum�ľ���
    clear disSupCen labCen
    
    % euclid�������W����ɫ�ռ䣬���е㶼��sal_lab��
    W_lab = normalize( DistanceZL(seg_vals, seg_vals, 'euclid') ); %�������W��ŷ����þ��룩
    sal_lab = exp( -theta*W_lab );%spum*spum�ľ���
    

    % Ranking
    spGraph = calSparseGraph( sal_lab, adjc, bord, spnum );%����ϡ�����
    W = sal_lab.*spGraph;%����ֻ�����������ڵĽڵ㸳ֵ(ֻ��ѡ�еļ����㸳ֵΪsal_lab)
    dd = sum(W,2);%�������D��ÿһ�н�����ͣ����Ϊ��������
    D = sparse(1:spnum,1:spnum,dd);%��Խ���(spnum*spnum)
    
    P = (D-alpha*W)\eye(spnum); %��D-alpha*W��-1��spnum*spnum��
    Sal = P*simSupCen;%��D-alpha*W��-1*SIM��spnum*cenNum��
    Sal = normalize( Sal' );%��һ��sal��cenNum*spnum��
    clear W dd D
    
 
  
    salSup = Sal.*(ones(spnum,1)*num_vals');%hij*nj(��˺ͳ˲�һ��������ĵ���Ƕ�Ӧλ�õ�ÿһ��Ԫ����ˣ���˵õ�����cenNum*spnum����)
    Isum = sum(salSup,2);%���sum��hij*nj��=Hi��������ͣ�����һ�����������õ��������г����ؿ��ÿһ��������Ӱ��ͣ�
    
    %%-----------������ܶ�-----------%%
    comSal = calDistribution(salSup,Isum,x_vals,y_vals,m,n,spnum,spnum);
    comVal= calfgDistributionSal(salSup,Isum,x_vals,y_vals,m,n,spnum,spnum);%������ܶȣ��Ǹ�cenNum*1����������  
    
     %%-----------���������ܶ�ͼд���ļ�����-----------%%
%     InitCom= zeros(m,n);
%     for k = 1:spnum
%           InitCom(inds{k}) = comVal(k);
%     end
%     mapstage=zeros(w(1),w(2));
%     mapstage(w(3):w(4),w(5):w(6))= InitCom;
%     mapstage=uint8(mapstage*255);
%     outname=[InitCom_folderName{M},'/', imnames(ii).name(1:end-4) '.jpg'];
%     imwrite(mapstage,outname);
    
    
    %%---------�����Խ��ܶ�Ϊ������������������-------%%

   X=zeros(spnum,1);
   Y=zeros(spnum,1);
   Obj_Com_Cen=zeros(spnum,1);
    for i=1:spnum
     X(i)=comVal(i).*x_vals(i);
     Y(i)=comVal(i).*y_vals(i);
    end
     Mean_X=sum(X)/sum(comVal);
     Mean_Y=sum(Y)/sum(comVal);
%      Obj_Cen=zeros(spnum,1);
     for i=1:spnum
     Obj_Com_Cen(i)=exp(-(x_vals(i)-Mean_X).^2/(2*(0.15*m)^2)-(y_vals(i)-Mean_Y).^2/(2*(0.15*n)^2));
     end
    clear salSup Isum 
   
    
    
     %%------------������ܶ�Ϊ���������������Լ����Ŷ�--------%%
     ComObj_vals=zeros(spnum,spnum);
       for i=1:spnum
           for j=1:spnum 
             ComObj_vals(i,j)=exp(-(comSal(i)-comSal(j)).^2);
           end
       end
       
       %%-----------------------ǰ����������ϡ�����Ĺ���---------%%
     fgProbComInd=comSal<0.1;
%      fgProbComInd=fgProbComInd';
     fgProbComInd=fgProbComInd*ones(1,spnum);
%      fgProbComInd=setdiff(spGraph,fgProbComInd);
     
     
      bgProbComInd=comSal>0.9;
%      fgProbComInd=fgProbComInd';
      bgProbComInd=bgProbComInd*ones(1,spnum);
%       bgProbComInd=setdiff(spGraph,bgProbComInd);    
       
       
       %%--------����������---------%%
        W0 =sal_lab.*spGraph;
        W1= normalize(ComObj_vals).*(fgProbComInd+ bgProbComInd);
        W2=W0+W1;
    dd = sum(W2,2);%�������D��ÿһ�н�����ͣ����Ϊ��������
    D = sparse(1:spnum,1:spnum,dd);%��Խ���(spnum*spnum)
    P = (D-alpha*W2)\eye(spnum); %��D-alpha*W��-1��spnum*spnum��
    
    
    %%-------�������յ��Խ��ܶ�Ϊ����������ͼ----------%%
      comVal=comVal.*Obj_Com_Cen;
    %%-------��ϸ����Ľ��ܶ�Ϊ����������ͼд���ļ�����----------%%   
     
      
      
      
     fgSal_NoDiff = normalize(sum(comVal*ones(1,spnum).*simSupCen,1));
%      fgSal_Com = normalize(sum(comVal*ones(1,spnum).*simSupCen,1));
     fgSal_Com = normalize(P*fgSal_NoDiff');

     
     
      salall_Compact = zeros(m,n);  
    for k = 1:spnum
        salall_Compact(inds{k}) = fgSal_Com(k);
%       salall_Compact(inds{k}) = CompactMap(k);
    end
%     clear k x_vals y_vals adjc finSal
%     clear num_vals seg_vals spnum inds


 
    mapstage4=zeros(w(1),w(2));
    mapstage4(w(3):w(4),w(5):w(6))=salall_Compact;
    mapstage4=uint8(mapstage4*255);
    outname4=[com_folderName{M},'/',imnames(ii).name(1:end-4) '.jpg'];
    imwrite(mapstage4,outname4);
  
     

    %% ---------------------- ����Աȶ� ---------------------%%  
   %ȡ�������½Ǳ�
    [~, index] = max(simSupCen,[],2);%ȡÿ�е����ֵ��Ҳ����ÿ�������غ��ĸ���������ƶ���󣬷��ص���ÿ�������ֵ���е�λ��(1*cenNum)
    bgIndex = unique([superpixels(1,:),superpixels(m,:),superpixels(:,1)',superpixels(:,n)']);%ȡ��������һ�б�ǵ�ֵ����ǩ��ֵ��
    bgCenInd = unique(index(bgIndex));%ȡ���߽��ϵı�����
    bgComInd = find(comVal == 0);%�ҳ����ڱ������½Ǳ�
    fgComInd = find(comVal ~= 0);%�ҳ�����ǰ�����½Ǳ�
    bgInd = unique([bgCenInd;bgComInd]);
    bgInd = setdiff( bgInd,fgComInd );%ȥ������ǰ���ĽǱ�

     %����Աȶȣ�ֻ�Ǻͱ����ĳ����ص����)
    Bg_sp_vals=[x_vals(bgInd) y_vals(bgInd)];
    Bg_seg_vals=seg_vals(bgInd,:);
    Sp_lab=normalize(DistanceZL(sp_vals, Bg_sp_vals, 'euclid'));
    Color_lab= normalize( DistanceZL(seg_vals,Bg_seg_vals, 'euclid'));
    Con_vals=(Color_lab).*exp( -2.5*(Sp_lab.^2));
     ConSal=normalize(fgSal_Com.*(sum(Con_vals,2)));
%      ConSal=normalize(fgSal_Com'.*(sum(Con_vals,2)));
%     ConSal_NoCom=normalize(sum(Con_vals,2));

    

     %%--------�����ԶԱȶ�Ϊ������������������----------%%
   X1=zeros(spnum,1);
   Y1=zeros(spnum,1);
   Obj_Con_Cen=zeros(spnum,1);
    for i=1:spnum
     X1(i)=ConSal(i).*x_vals(i);
     Y1(i)=ConSal(i).*y_vals(i);
    end
     Mean_X1=sum(X1)/(sum(ConSal));
     Mean_Y1=sum(Y1)/(sum(ConSal));
     for i=1:spnum
     Obj_Con_Cen(i)=exp(-(x_vals(i)-Mean_X1).^2/(2*(0.20*m)^2)-(y_vals(i)-Mean_Y1).^2/(2*(0.20*n)^2));
     end
   



     
    
%%---------------------------����-------------------%%
    %������ڶԱȶȵ����������Լ����Ŷ�
      ConObj_vals=zeros(spnum,spnum);
%      C_vals=zeros(spnum,spnum);
       for i=1:spnum
%            C=sum(sal_lab,2)./spnum;
           for j=1:spnum
%              Obj_Con_vals(i,j)=(0.6*ConSal(i)*ConSal(j)+0.4*Obj_Con_Cen(i)* Obj_Con_Cen(j))/2; 
%                Obj_Con_vals(i,j)=ConSal(i)*ConSal(j);
                ConObj_vals(i,j)=exp(-(ConSal(i)- ConSal(j)).^2);
%              C_vals(i,j)=(1-C(i))*(1-C(j)');
           end
       end 
    W3= sal_lab.*spGraph;
    dd1 = sum(W3,2);%�������D��ÿһ�н�����ͣ����Ϊ��������
    D1 = sparse(1:spnum,1:spnum,dd1);%��Խ���(spnum*spnum)
    P1 = (D1-alpha*W3)\eye(spnum); %��D-alpha*W��-1��spnum*spnum) 

 %%-------�������յ��ԶԱȶ�Ϊ����������ͼ----------%%
    ConSal=ConSal.* Obj_Con_Cen;
    fgSal_Con=normalize(P1*ConSal);  

  
     salall_Contrast = zeros(m,n);
    for k = 1:spnum
%         salall_Contrast(inds{k}) =   ContrastMap(k);
 salall_Contrast(inds{k}) =   fgSal_Con(k);
%  salall_Contrast(inds{k}) = ConSal(k);
    end
%     clear k x_vals y_vals adjc finSal
%     clear num_vals seg_vals
%   
    mapstage7=zeros(w(1),w(2));
    mapstage7(w(3):w(4),w(5):w(6))=salall_Contrast;
    mapstage7=uint8(mapstage7*255);
    outname7=[con_folderName{M},'/', imnames(ii).name(1:end-4) '.jpg'];
    imwrite(mapstage7,outname7);
    

    
      %%----------------�ϲ�����---------------------%%
    %���
    mapstage8=zeros(w(1),w(2));
    mapstage8(w(3):w(4),w(5):w(6))=normalize(salall_Contrast+salall_Compact);
    mapstage8=uint8(mapstage8*255);
    outname8=[sal_folderName{M},'/', imnames(ii).name(1:end-4) '.jpg'];
    imwrite(mapstage8,outname8);
%     clear mapstage6 outname6 w salall    
     
    

    
end
sal_folderName1=sal_folderName{1};
sal_folderName2=sal_folderName{2};
sal_folderName3=sal_folderName{3};
sal_folderName4=sal_folderName{4};
sal_folderName5=sal_folderName{5};

imgRoot1 = [sal_folderName1,'/'];
imgRoot2 = [sal_folderName2,'/'];
imgRoot3 = [sal_folderName3,'/'];
imgRoot4 = [sal_folderName4,'/'];
imgRoot5 = [sal_folderName5,'/'];


imnames1 = dir([ imgRoot1 '*' 'jpg']);


    imName1 = [ imgRoot1 imnames1(ii).name(1:end-4)  '.jpg' ];  
    imName2 = [ imgRoot2 imnames1(ii).name(1:end-4)  '.jpg' ];
    imName3 = [ imgRoot3 imnames1(ii).name(1:end-4)  '.jpg' ];
    imName4 = [ imgRoot4 imnames1(ii).name(1:end-4)  '.jpg' ];  
    imName5 = [ imgRoot5 imnames1(ii).name(1:end-4)  '.jpg' ];
    
    input_im1=double(imread(imName1));
    input_im2=double(imread(imName2));
    input_im3=double(imread(imName3));
    input_im4=double(imread(imName4));
    input_im5=double(imread(imName5));
    
    % avoid that saliency values equal zero
    input_im1=normalize_1(input_im1,0);  
    input_im2=normalize_1(input_im2,0);
    input_im3=normalize_1(input_im3,0);
    input_im4=normalize_1(input_im4,0);  
    input_im5=normalize_1(input_im5,0);

    [m,n]=size(input_im1);
    M=5;
    S_M=cell(1,M);
    %��ÿһ�������ֵ����Ԫ����
    for i=1:M               
         if i==1
            S_M{i}=input_im1;
            else if i==2
                 S_M{i}=input_im2;
              else if i==3
                      S_M{i}=input_im3;
                 else if i==4
                         S_M{i}=input_im4;
                     else  
                         S_M{i}=input_im5;
                     end
                  end
                end
         end
    end 

    guassianTemplate=cell(1,5);
   guassOptimizeResult=cell(1,5); 
%    FinalMap=zeros(m,n);
%�Ը��´���Ϊ3��4�����飩
Lap=4;
nLapMap=cell(1,Lap);
Map=cell(1,5);
for N=1:5 
guassianTemplate{N} = calOptimizedGuassTemplate(S_M{N},guassSigmaRatio,[m n]);
guassOptimizeResult{N} = guassianTemplate{N}.*S_M{N};
% outname9=[sal_Gauss_folderName{N},'/' imnames(ii).name(1:end-4)  '.jpg'];
% imwrite(guassOptimizeResult{N},outname9);

nLapMap{1}= guassOptimizeResult{N};
 for nLap=2:Lap
          nLapMap{nLap}=nLapMap{nLap-1}+sign(nLapMap{nLap-1}-graythresh(nLapMap{nLap-1})).*0.08;
end   
   Map{N}= nLapMap{Lap}; 
%   Map=normalize(Map);
%  outname10=[ SGX_sal_folderName{N} '/'  imnames(ii).name(1:end-4)  '_Our.png'];
%   imwrite(Map{N},outname10); 
  
end
    
    S_N2=zeros(m,n);
    for i=1:M
        S_N2=S_N2+Map{i};
%        S_N2=S_N2+S_M{M};
%       S_N2=S_N2+guassOptimizeResult{N};
    end
   
 S_Normalize=graythresh(normalize(S_N2));
    coda_2_sign=cell(1,5);   
    N2=5;
    for lap=1:N2
        for i=1:5               
           coda_2_sign{i}=sign(Map{i}-S_Normalize);
        end
      for i=1:5      
            for j=1:5
           Map{i}=Map{i}+coda_2_sign{i}.*0.08;
            end
       end
    end
    

  finalMap=zeros(m,n);
    for i=1:5
        finalMap=finalMap+Map{i};
    end

    finalMap=normalize( finalMap);
%     finalMap=normalize(S_N2);
    outname=[finaldir  imnames(ii).name(1:end-4)  '_Our.png'];
    imwrite(finalMap,outname); 
  



end
end   