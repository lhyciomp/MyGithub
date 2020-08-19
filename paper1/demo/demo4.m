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
       %%-------------------设置生成放置图片的文件夹------------------%%
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
    comm=['SLICSuperpixelSegmentation' ' ' imname ' ' int2str(20) ' ' int2str(spnumber(M)) ' ' sup_folderName{M}];%使用SLIC软件，将inname分割，并存入supdir文件夹中
    system(comm);    %SLIC分割
    spname=[sup_folderName{M} imnames(ii).name(1:end-4)  '.dat'];%分割后的SLIC.dat文件
    superpixels=ReadDAT([m,n],spname); % superpixel label matrix，读取分割后的图片中超像素块的信息，保存的应该是标签号
    spnum=max(superpixels(:));% the actual superpixel number获取超像素的个数（：）的作用是将每列合并成一个列向量，最大标签的标号就代表超像素的个数
    [adjc ,bord] = calAdjacentMatrix(superpixels,spnum); 

    clear comm spname
    
    
    % compute the feature (mean color in lab color space) for each superpixels    
    rgb_vals = zeros(spnum,3);%生成一个spum*3的零矩阵
    inds=cell(spnum,1);%生成一个spum*1的空单元矩阵
    [x, y] = meshgrid(1:1:n, 1:1:m);%网格采样点（m*n）
    x_vals = zeros(spnum,1);%生成一个spum*1的全零矩阵；
    y_vals = zeros(spnum,1);
    num_vals = zeros(spnum,1);
    for i=1:spnum
        inds{i}=find(superpixels==i);%找出矩阵第i个超像素块，返回像素块的单下标
        num_vals(i) = length(inds{i});%获取超像素块中的像素数目
        rgb_vals(i,:) = mean(input_vals(inds{i},:),1);%对超像素块i中的像素RGB值取平均（对列做平均）
        x_vals(i) = sum(x(inds{i}))/num_vals(i);%取出超像素中的每个像素的x值，并做平均（超像素块i中心坐标x）
        y_vals(i) = sum(y(inds{i}))/num_vals(i);%取出超像素中的每个像素的y值，并做平均（超像素块i中心坐标y）
    end  
    seg_vals = colorspace('Lab<-', rgb_vals);%将rgb颜色空间转换为lab颜色空间
    sp_vals=[x_vals y_vals];%spnum*2的矩阵，是所有超像素块的空间中心坐标
    clear i x y input_vals rgb_vals
    
    
    %计算相似度矩阵A（sim）
    disSupCen = normalize( DistanceZL(seg_vals, seg_vals, 'euclid'), 'column' );
    simSupCen = exp( -theta*disSupCen );%spnum*cenNum的矩阵
    clear disSupCen labCen
    
    % euclid计算矩阵W（颜色空间，所有点都是sal_lab）
    W_lab = normalize( DistanceZL(seg_vals, seg_vals, 'euclid') ); %计算矩阵W（欧几里得距离）
    sal_lab = exp( -theta*W_lab );%spum*spum的矩阵
    

    % Ranking
    spGraph = calSparseGraph( sal_lab, adjc, bord, spnum );%构造稀疏矩阵
    W = sal_lab.*spGraph;%这是只给属于邻域内的节点赋值(只有选中的几个点赋值为sal_lab)
    dd = sum(W,2);%构造矩阵D（每一行进行求和，结果为列向量）
    D = sparse(1:spnum,1:spnum,dd);%求对角线(spnum*spnum)
    
    P = (D-alpha*W)\eye(spnum); %（D-alpha*W）-1（spnum*spnum）
    Sal = P*simSupCen;%（D-alpha*W）-1*SIM（spnum*cenNum）
    Sal = normalize( Sal' );%归一化sal（cenNum*spnum）
    clear W dd D
    
 
  
    salSup = Sal.*(ones(spnum,1)*num_vals');%hij*nj(点乘和乘不一样，这里的点乘是对应位置的每一个元素相乘，因此得到的是cenNum*spnum矩阵)
    Isum = sum(salSup,2);%求和sum（hij*nj）=Hi（对行求和，生成一个列向量，得到的是所有超像素块对每一个聚类块的影响和）
    
    %%-----------计算紧密度-----------%%
    comSal = calDistribution(salSup,Isum,x_vals,y_vals,m,n,spnum,spnum);
    comVal= calfgDistributionSal(salSup,Isum,x_vals,y_vals,m,n,spnum,spnum);%计算紧密度（是个cenNum*1的列向量）  
    
     %%-----------将初步紧密度图写入文件夹中-----------%%
%     InitCom= zeros(m,n);
%     for k = 1:spnum
%           InitCom(inds{k}) = comVal(k);
%     end
%     mapstage=zeros(w(1),w(2));
%     mapstage(w(3):w(4),w(5):w(6))= InitCom;
%     mapstage=uint8(mapstage*255);
%     outname=[InitCom_folderName{M},'/', imnames(ii).name(1:end-4) '.jpg'];
%     imwrite(mapstage,outname);
    
    
    %%---------计算以紧密度为基础的物体中心先验-------%%

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
   
    
    
     %%------------计算紧密度为基础的物体先验以及置信度--------%%
     ComObj_vals=zeros(spnum,spnum);
       for i=1:spnum
           for j=1:spnum 
             ComObj_vals(i,j)=exp(-(comSal(i)-comSal(j)).^2);
           end
       end
       
       %%-----------------------前景及背景的稀疏矩阵的构造---------%%
     fgProbComInd=comSal<0.1;
%      fgProbComInd=fgProbComInd';
     fgProbComInd=fgProbComInd*ones(1,spnum);
%      fgProbComInd=setdiff(spGraph,fgProbComInd);
     
     
      bgProbComInd=comSal>0.9;
%      fgProbComInd=fgProbComInd';
      bgProbComInd=bgProbComInd*ones(1,spnum);
%       bgProbComInd=setdiff(spGraph,bgProbComInd);    
       
       
       %%--------传播矩阵构造---------%%
        W0 =sal_lab.*spGraph;
        W1= normalize(ComObj_vals).*(fgProbComInd+ bgProbComInd);
        W2=W0+W1;
    dd = sum(W2,2);%构造矩阵D（每一行进行求和，结果为列向量）
    D = sparse(1:spnum,1:spnum,dd);%求对角线(spnum*spnum)
    P = (D-alpha*W2)\eye(spnum); %（D-alpha*W）-1（spnum*spnum）
    
    
    %%-------计算最终的以紧密度为基础的显著图----------%%
      comVal=comVal.*Obj_Com_Cen;
    %%-------将细化后的紧密度为基础的显著图写入文件夹中----------%%   
     
      
      
      
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
  
     

    %% ---------------------- 计算对比度 ---------------------%%  
   %取背景的下角标
    [~, index] = max(simSupCen,[],2);%取每行的最大值，也就是每个超像素和哪个聚类块相似度最大，返回的是每行中最大值的列的位置(1*cenNum)
    bgIndex = unique([superpixels(1,:),superpixels(m,:),superpixels(:,1)',superpixels(:,n)']);%取出的是这一列标记的值（标签的值）
    bgCenInd = unique(index(bgIndex));%取出边界上的背景点
    bgComInd = find(comVal == 0);%找出属于背景的下角标
    fgComInd = find(comVal ~= 0);%找出属于前景的下角标
    bgInd = unique([bgCenInd;bgComInd]);
    bgInd = setdiff( bgInd,fgComInd );%去除属于前景的角标

     %计算对比度（只是和背景的超像素点相比)
    Bg_sp_vals=[x_vals(bgInd) y_vals(bgInd)];
    Bg_seg_vals=seg_vals(bgInd,:);
    Sp_lab=normalize(DistanceZL(sp_vals, Bg_sp_vals, 'euclid'));
    Color_lab= normalize( DistanceZL(seg_vals,Bg_seg_vals, 'euclid'));
    Con_vals=(Color_lab).*exp( -2.5*(Sp_lab.^2));
     ConSal=normalize(fgSal_Com.*(sum(Con_vals,2)));
%      ConSal=normalize(fgSal_Com'.*(sum(Con_vals,2)));
%     ConSal_NoCom=normalize(sum(Con_vals,2));

    

     %%--------计算以对比度为基础的物体中心先验----------%%
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
   



     
    
%%---------------------------传播-------------------%%
    %计算基于对比度的物体先验以及置信度
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
    dd1 = sum(W3,2);%构造矩阵D（每一行进行求和，结果为列向量）
    D1 = sparse(1:spnum,1:spnum,dd1);%求对角线(spnum*spnum)
    P1 = (D1-alpha*W3)\eye(spnum); %（D-alpha*W）-1（spnum*spnum) 

 %%-------计算最终的以对比度为基础的显著图----------%%
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
    

    
      %%----------------合并二者---------------------%%
    %相加
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
    %把每一层的显著值读入元胞中
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
%自更新次数为3（4的数组）
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