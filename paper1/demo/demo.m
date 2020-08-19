function demo

clear all;
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
tsdir='./ts/';%存储纹理抑制后的图片的文件夹
% mkdir(supdir);
mkdir(saldir);
% mkdir(compdir);
mkdir(tsdir);
% mkdir(condir);
imnames=dir([imgRoot '*' '.jpg']);
 sup_folderName=cell(1,5);
 com_folderName=cell(1,5);
 con_folderName=cell(1,5);
 sal_folderName=cell(1,5);
 Obj_Cen=cell(1,5);
 Obj_Cen_Con=cell(1,5);
 fgSal_Com=cell(1,5);

for ii=1:length(imnames)   
for M=1:length(spnumber)
    sup_folderName{M}=['./superpixels',num2str(M),'/'];
    com_folderName{M}=['./salCompact',num2str(M)];
    con_folderName{M}=['./salContrast',num2str(M)];
    sal_folderName{M}=['./saliencymap',num2str(M)];
  
    mkdir(sup_folderName{M});
    mkdir(com_folderName{M});
    mkdir(con_folderName{M});
    mkdir(sal_folderName{M});
    disp(ii);
    imname = [imgRoot imnames(ii).name]; 
    [input_im, w]=removeframe(imname);% run a pre-processing to remove the image frame 
    
    [m,n,k] = size(input_im);    
    input_vals=reshape(input_im, m*n, k);
    clear k
    
    %纹理抑制
%     S=imread(imname);
%     S=L0Smoothing(S,0.0035);
%     outname=[tsdir imnames(ii).name(1:end-4) '.jpg'];
%     imwrite(S,outname);
%     imname = [tsdir imnames(ii).name]; 
%     [input_im, w]=removeframe(imname);% run a pre-processing to remove the image frame 
%     [m,n,k] = size(input_im);    
%     input_vals=reshape(input_im, m*n, k);
%     clear k
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
    
    % k-meams
%     labCen = zl_kmeans(cenNum, seg_vals, options);%聚类
    
    %计算相似度矩阵A（sim）
    disSupCen = normalize( DistanceZL(seg_vals, seg_vals, 'euclid'), 'column' );
    simSupCen = exp( -theta*disSupCen );%spnum*cenNum的矩阵
    clear disSupCen labCen
    
    % euclid计算矩阵W（颜色空间，所有点都是sal_lab）
    W_lab = normalize( DistanceZL(seg_vals, seg_vals, 'euclid') ); %计算矩阵W（欧几里得距离）
    sal_lab = exp( -theta*W_lab );%spum*spum的矩阵
    

     % euclid计算矩阵W（尺寸空间，所有点都是sal_lab）
%     Sp_lab = normalize( DistanceZL(sp_vals, sp_vals, 'euclid') ); %计算矩阵W（欧几里得距离）
%     SpSal_lab = exp( -theta*Sp_lab );%spum*spum的矩阵
    
    %合并二者
%     sal_lab=ColorSal_lab+ SpSal_lab;
%     sal_lab=sal_lab.*(1-C_vals);

    % Ranking
    spGraph = calSparseGraph( sal_lab, adjc, bord, spnum );%构造稀疏矩阵
    W = sal_lab.*spGraph;%这是只给属于邻域内的节点赋值(只有选中的几个点赋值为sal_lab)
    dd = sum(W,2);%构造矩阵D（每一行进行求和，结果为列向量）
    D = sparse(1:spnum,1:spnum,dd);%求对角线(spnum*spnum)
    
    P = (D-alpha*W)\eye(spnum); %（D-alpha*W）-1（spnum*spnum）
    Sal = P*simSupCen;%（D-alpha*W）-1*SIM（spnum*cenNum）
    Sal = normalize( Sal' );%归一化sal（cenNum*spnum）
    clear W dd D
    
   weight = calPostprocessingWeight(sal_lab,adjc,spnum);
  
    salSup = Sal.*(ones(spnum,1)*num_vals');%hij*nj(点乘和乘不一样，这里的点乘是对应位置的每一个元素相乘，因此得到的是cenNum*spnum矩阵)
    Isum = sum(salSup,2);%求和sum（hij*nj）=Hi（对行求和，生成一个列向量，得到的是所有超像素块对每一个聚类块的影响和）
    comSal = calDistribution(salSup,Isum,x_vals,y_vals,m,n,spnum,spnum);
    comVal= calfgDistributionSal(salSup,Isum,x_vals,y_vals, weight,m,n,spnum,spnum);%计算紧密度（是个cenNum*1的列向量）
    for i=1:spnum
     X=comVal(i).*x_vals(i);
     Y=comVal(i).*y_vals(i);
    end
     Mean_X=sum(X)/sum(comVal);
     Mean_Y=sum(Y)/sum(comVal);
%      Obj_Cen=zeros(spnum,1);
     for i=1:spnum
     Obj_Cen{M}(i)=exp(-(x_vals(i)-Mean_X).^2/(2*(0.25*m)^2)-(y_vals(i)-Mean_Y).^2/(2*(0.25*n)^2));
     end
    clear salSup Isum
%  %计算物体先验以及置信度
%      Obj_vals=zeros(spnum,spnum);
%      C_vals=zeros(spnum,spnum);
%        for i=1:spnum
%            C=sum(sal_lab,2)./spnum;
%            for j=1:spnum
%              Obj_vals(i,j)= comVal(i)*comVal(j); 
%              C_vals(i,j)=(1-C(i))*(1-C(j)');
% %            C_vals(i,j)=1- sal_lab(i,j);
%            end
%        end
%     W = normalize(0.7*Obj_vals+sal_lab.*(1-C_vals)).*spGraph;
%     W = sal_lab.*spGraph;
%     dd = sum(W,2);%构造矩阵D（每一行进行求和，结果为列向量）
%     D = sparse(1:spnum,1:spnum,dd);%求对角线(spnum*spnum)
%     P = (D-alpha*W)\eye(spnum); %（D-alpha*W）-1（spnum*spnum）
    

     %取背景的下角标
    [~, index] = max(simSupCen,[],2);%取每行的最大值，也就是每个超像素和哪个聚类块相似度最大，返回的是每行中最大值的列的位置(1*cenNum)
    bgIndex = unique([superpixels(1,:),superpixels(m,:),superpixels(:,1)',superpixels(:,n)']);%取出的是这一列标记的值（标签的值）
    bgCenInd = unique(index(bgIndex));%取出边界上的背景点
    bgComInd = find(comVal == 0);%找出属于背景的下角标
    fgComInd = find(comVal ~= 0);%找出属于前景的下角标
    bgInd = unique([bgCenInd;bgComInd]);
    bgInd = setdiff( bgInd,fgComInd );%去除属于前景的角标
     
    %求前景权重
%     colDistM = GetDistanceMatrix(seg_vals);
%     [clipVal, geoSigma] = EstimateDynamicParas(adjc, colDistM);
%     fgProb = EstimateBgProb( colDistM, adjc, bgIndex, clipVal, geoSigma);
    
    %Ranking
     %计算物体先验以及置信度
     Obj_vals=zeros(spnum,spnum);
     C_vals=zeros(spnum,spnum);
     ComObj_vals=zeros(spnum,spnum);
       for i=1:spnum
           C=sum(sal_lab,2)./spnum;
           for j=1:spnum
%              Obj_vals(i,j)= comVal(i)*comVal(j)*fgProb(i)*fgProb(j); 
             Obj_vals(i,j)= 0.6*comVal(i)*comVal(j);
             ComObj_vals(i,j)=exp(-0.1*(comSal(i)-comSal(j)).^2);
             C_vals(i,j)=(1-C(i))*(1-C(j)');
%            C_vals(i,j)=1- sal_lab(i,j);
           end
       end
%         Obj_vals=normalize(exp(Obj_vals));
%      a=(1:spnum)';
%       comVal_NoMean=comVal_NoMean*comVal_NoMean';


%%前景及背景的稀疏矩阵的构造


     fgProbComInd=comSal<0.1;
%      fgProbComInd=fgProbComInd';
     fgProbComInd=fgProbComInd*ones(1,spnum);
     fgProbComInd=setdiff(spGraph,fgProbComInd);
     
     
      bgProbComInd=comSal>0.9;
%      fgProbComInd=fgProbComInd';
      bgProbComInd=bgProbComInd*ones(1,spnum);
      bgProbComInd=setdiff(spGraph,bgProbComInd);
     
%      bgProbComInd=bgProb==0.9;
%      Sparse_Fg=zeros(spnum,spnum);
%     for i=1:spnum
%         a=1:spnum;
%         b=i.*ones(1,spnum);
%       a(sub2ind(size(a),b,a))=fgProbComInd; 
%        Sparse_Fg(i,:)=a;
%     end


%       W = normalize(Obj_vals+sal_lab.*(1-C_vals)).*(spGraph+ fgProbComInd+ bgProbComInd);
 W1 = normalize(Obj_vals+sal_lab.*(1-C_vals)).*spGraph;
 W2= normalize(ComObj_vals).*(fgProbComInd+ bgProbComInd);
 W=W1+W2;
% W = normalize(sal_lab.*(1-C_vals)).*spGraph;
    dd = sum(W,2);%构造矩阵D（每一行进行求和，结果为列向量）
    D = sparse(1:spnum,1:spnum,dd);%求对角线(spnum*spnum)
    P = (D-alpha*W)\eye(spnum); %（D-alpha*W）-1（spnum*spnum）
    
    
%     simSupCen = simSupCen';%转置后变成了CenNum*spnum
%     bgSal = 1-normalize(simSupCen(bgInd(1),:));%属于背景的聚类块的相似值Sik(行向量)
%     for i = 2:length(bgInd)
%         bgSal = bgSal.*( 1-normalize(simSupCen(bgInd(i),:)) );%相乘
%     end
%     bgSal = normalize(bgSal);
%     bgSal = normalize(P*bgSal');

%     fgSal = normalize(sum(comVal*ones(1,spnum).*simSupCen,1));
%      comVal=comVal.* Obj_Cen;
     fgSal_NoDiff = normalize(sum(comVal*ones(1,spnum).*simSupCen,1));
     fgSal_Com{M} = normalize(P*fgSal_NoDiff');
%   fgSal = normalize( sum(weight.*padarray(fgSal',[spnum-1 0],'replicate','post'),2) );
%    clear P index bgIndex bgCenInd bgComInd fgComInd bgInd simSupCen
    
%     finSal = normalize( 0.6*fgSal + 0.4*bgSal);
%     clear superpixels weight W_lab sal_lab Sal simSupCen comVal fgSal bgSal
    
%     salall_Compact = zeros(m,n);
%     for k = 1:spnum
%         salall_Compact(inds{k}) = fgSal(k);
%     end
% %     clear k x_vals y_vals adjc finSal
% %     clear num_vals seg_vals spnum inds
%     
%     mapstage=zeros(w(1),w(2));
%     mapstage(w(3):w(4),w(5):w(6))=salall_Compact;
%     mapstage=uint8(mapstage*255);
%     outname=[com_folderName{M},'/',imnames(ii).name(1:end-4) '.jpg'];
%     imwrite(mapstage,outname);
    
%     clear mapstage1 outname w salall m n
%     clear mapstage1 outname  salall  
    %% ---------------------- 计算对比度 ---------------------%%  
    %第一种计算方法（只和背景的超像素节点相比）
%  Bg_sp_vals=[x_vals(bgInd) y_vals(bgInd)];
% con_vals=zeros(spnum,length(bgInd));
% % Sp_lab=zeros(bgInd,1);
% for i=1:spnum
%     for j=1:length(bgInd)
% %      Bg_sp_vals(j)=[x_vals(bgInd(j)) y_vals(bgInd(j))];
%        Sp_lab = normalize( DistanceZL(sp_vals(i,:), sp_vals(bgInd(j),:), 'euclid') ); %计算空间尺度上的权重（欧几里得距离）
%        Color_lab = normalize( DistanceZL(seg_vals(i,:), seg_vals(bgInd(j),:), 'euclid') ); 
%        con_vals(i,j)=exp( -theta*Sp_lab)*Color_lab;
%     end
% end
% conSal=normalize(sum(con_vals,2)); 
% conSal = normalize(P*conSal);

     %计算对比度（全局对比度)
    All_Sp_lab=normalize(DistanceZL(sp_vals, sp_vals, 'euclid'));
    All_Color_lab= normalize( DistanceZL(seg_vals,seg_vals, 'euclid'));
   All_Con_vals=All_Color_lab.*exp( -2.5*(All_Sp_lab.^2));
%     ConSal=normalize(sum(Con_vals,2));
    All_ConSal=normalize(fgSal.*(sum(All_Con_vals,2)));
%     ConSal=calContrastSal(ConSal);




     %计算对比度（第二种计算方法，只是和背景的超像素点相比)
%     Bg_sp_vals=[x_vals(bgInd) y_vals(bgInd)];
%     Bg_seg_vals=seg_vals(bgInd,:);
%     Sp_lab=normalize(DistanceZL(sp_vals, Bg_sp_vals, 'euclid'));
%     Color_lab= normalize( DistanceZL(seg_vals,Bg_seg_vals, 'euclid'));
%     Con_vals=(Color_lab).*exp( -2.5*(Sp_lab.^2));
% %     ConSal=normalize(sum(Con_vals,2));
% %     ConSal=normalize(fgSal.*(sum(Con_vals,2)));
%  ConSal=normalize(sum(Con_vals,2));
% %      Consal=calContrastSal(ConSal);
 

for i=1:spnum
     X=All_ConSal(i).*x_vals(i);
     Y=All_ConSal(i).*y_vals(i);
 end
     Mean_X=sum(X)/sum(All_ConSal);
     Mean_Y=sum(Y)/sum(All_ConSal);
%      Obj_Cen_Con=zeros(spnum,1);
     for i=1:spnum
     Obj_Cen_Con{M}=exp(-(x_vals(i)-Mean_X).^2/(2*(0.25*m)^2)-(y_vals(i)-Mean_Y).^2/(2*(0.25*n)^2));
     end
     
%   %计算对比度（和邻居的超像素相比）
%   Neb=zeros(spnum,spnum);
%   for i=1:spnum
%       Neb(i,:)=adjc(i,:).*(1:spnum);
%       NebIndex=find(Neb(i,:));
%    end
%       con_vals=zeros(spnum,length(NebIndex));
%    for i=1:spnum
%      for j=1:length(NebIndex)
% %       Neb_sp_vals(j)=[x_vals(NebIndex(j)) y_vals(NebIndex(j))];
%         Neb_Sp_lab = normalize( DistanceZL(sp_vals(i,:), sp_vals(NebIndex(j),:), 'euclid') ); %计算空间尺度上的权重（欧几里得距离）
%         Neb_Color_lab = normalize( DistanceZL(seg_vals(i,:), seg_vals(NebIndex(j),:), 'euclid') ); 
%         con_vals(i,:)=exp( -2.5*(Neb_Sp_lab.^2))*Neb_Color_lab.^2;
%      end
%    end
%      ConSal=normalize(sum(con_vals,2)); 
% %      ConSal=calContrastSal(ConSal);



%%--------------------------------传播-------------------%%
    %计算物体先验以及置信度
%      ObjCon_vals=zeros(spnum,spnum);
%      C_vals=zeros(spnum,spnum);
     ConObj_vals=zeros(spnum,spnum);
       for i=1:spnum
%            C=sum(sal_lab,2)./spnum;
           for j=1:spnum
%              ObjCon_vals(i,j)=0.6*ConSal(i)*ConSal(j); 
             ConObj_vals(i,j)=exp(-(All_ConSal(i)-All_ConSal(j)).^2);
%              C_vals(i,j)=(1-C(i))*(1-C(j)');
%            C_vals(i,j)=1- sal_lab(i,j);
           end
       end
%        ObjCon_vals=normalize(exp(ObjCon_vals));


%背景及前景稀疏矩阵的构造
%       [~,Sparse_FgIndex] = max(ObjCon_vals,[],2);
%       [~,Sparse_BgIndex]=min(ObjCon_vals,[],2);
%       a = 1:spnum;
%       ConSparse_Fg = sparse([a';Sparse_FgIndex],[Sparse_FgIndex;a'], ...
%       [ones(spnum,1);ones(spnum,1)],spnum,spnum);
%       ConSparse_Bg = sparse([a';Sparse_BgIndex],[Sparse_BgIndex;a'], ...
%       [ones(spnum,1);ones(spnum,1)],spnum,spnum);
%     fgProbConInd=ConSal>graythresh(ConSal);
% %      fgProbComInd=fgProbComInd';
%      fgProbConInd=fgProbConInd*ones(1,spnum);
% %      fgProbConInd=setdiff(spGraph,fgProbConInd);
%      
%      
%       bgProbConInd=ConSal<graythresh(ConSal);
% %      fgProbComInd=fgProbComInd';
%       bgProbConInd=bgProbConInd*ones(1,spnum);
% %       bgProbConInd=setdiff(spGraph,bgProbConInd);

    fgProbConInd=All_ConSal<0.3;
%      fgProbComInd=fgProbComInd';
     fgProbConInd=fgProbConInd*ones(1,spnum);
     fgProbConInd=setdiff(spGraph,fgProbConInd);
     
     
      bgProbConInd=All_ConSal>0.7;
%      fgProbComInd=fgProbComInd';
      bgProbConInd=bgProbConInd*ones(1,spnum);
      bgProbConInd=setdiff(spGraph,bgProbConInd);

  
%      W= normalize(ObjCon_vals+sal_lab.*(1-C_vals)).*spGraph;
      W1= normalize(sal_lab).*spGraph;
     W2=normalize(ConObj_vals).*(fgProbConInd+ bgProbConInd);
     W=W1+W2;
%   W = normalize(sal_lab.*(1-C_vals)).*spGraph;
%     W = sal_lab.*spGraph;
    dd = sum(W,2);%构造矩阵D（每一行进行求和，结果为列向量）
    D = sparse(1:spnum,1:spnum,dd);%求对角线(spnum*spnum)
    P = (D-alpha*W)\eye(spnum); %（D-alpha*W）-1（spnum*spnum) 
%     ConSal=normalize((ConSal+ All_ConSal)/2);
    ConSal=All_ConSal.* Obj_Cen;
    ConSal=normalize(P*ConSal);
    
    salall_Contrast = zeros(m,n);
    for k = 1:spnum
        salall_Contrast(inds{k}) = ConSal(k);
    end
    clear k x_vals y_vals adjc finSal
    clear num_vals seg_vals inds
    
    mapstage1=zeros(w(1),w(2));
    mapstage1(w(3):w(4),w(5):w(6))=salall_Contrast;
    mapstage1=uint8(mapstage1*255);
    outname1=[con_folderName{M},'/', imnames(ii).name(1:end-4) '.jpg'];
    imwrite(mapstage1,outname1);
    %%----------------合并二者---------------------%%
    %相加
%     mapstage2=zeros(w(1),w(2));
%     mapstage2(w(3):w(4),w(5):w(6))=normalize(salall_Compact+salall_Contrast);
%     mapstage2=uint8(mapstage2*255);
%     outname2=[sal_folderName{M},'/', imnames(ii).name(1:end-4) '.jpg'];
%     imwrite(mapstage2,outname2);
%     clear mapstage1 outname1 w salall m n 
end

 SM_Cen=zeros(spnum,1);
for M=1:5
    for i=1:spnum
    SM_Cen(i)=SM_Cen(i)+fgSal_Com{M}(i).* Obj_Cen{M}(i);  
    end
end
SM_Cen=SM_Cen/5;
ComMap=SM_Cen.*fgSal_Com;
    salall_Compact = zeros(m,n);
    for k = 1:spnum
        salall_Compact(inds{k}) = ComMap(k);
    end
%     clear k x_vals y_vals adjc finSal
%     clear num_vals seg_vals spnum inds
    
    mapstage=zeros(w(1),w(2));
    mapstage(w(3):w(4),w(5):w(6))=salall_Compact;
    mapstage=uint8(mapstage*255);
    outname=[com_folderName{M},'/',imnames(ii).name(1:end-4) '.jpg'];
    imwrite(mapstage,outname);

sal_folderName1=sal_folderName{1};
sal_folderName2=sal_folderName{2};
sal_folderName3=sal_folderName{3};
sal_folderName4=sal_folderName{4};
sal_folderName5=sal_folderName{5};

imgRoot1 = [sal_folderName1,'/'];
imgRoot2 = [sal_folderName2,'/'];
imgRoot3 = [sal_folderName3,'/'];
imgRoot4 = [sal_folderName4,'/'];
imgRoot5 = [sal_folderName5,'/']; % the input saliency maps paths
% saldir = './saliencymap_MCA/';% the output path of the saliency map
% mkdir(saldir);

imnames1 = dir([ imgRoot1 '*' 'jpg']);
imnames2 = dir([ imgRoot2 '*' 'jpg']);
imnames3 = dir([ imgRoot3 '*' 'jpg']);
imnames4 = dir([ imgRoot4 '*' 'jpg']);
imnames5 = dir([ imgRoot5 '*' 'jpg']);

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
    
 SM=zeros(5,5);
 for i=1:5
     for j=1:5
     SM(i,j)=sum(sum(exp(-(S_M{i}-S_M{j}).^2)),2);
     end
 end
  SM_final=sum(SM,2);
 index= find(min(SM_final));
 SM_final(index)=1;
SMap=cell(1,5);
  for i=1:5
 SMap{i}=SM_final(i)* S_M{i}; 
  end
  S_N2=zeros(m,n);
  for i=1:5
S_N2= S_N2+S_M{i}; 
  end 
  S_N2=normalize( S_N2);
  outname3=[saldir imnames(ii).name(1:end-4)  '.jpg'];
  imwrite(S_N2,outname3);
end
end
% Multilayer ( sal_folderName);
% FinalSaliencyMap=Multilayer (sal_folderName);
% outname3=[saldir imnames(ii).name(1:end-4)  '.jpg'];
% imwrite(FinalSaliencyMap,outname3);



