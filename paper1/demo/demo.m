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
       %%-------------------�������ɷ���ͼƬ���ļ���------------------%%
% supdir='./superpixels/';
% compdir='./salCompact/';
% condir='./salContrast/';
tsdir='./ts/';%�洢�������ƺ��ͼƬ���ļ���
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
    
    %��������
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
    
    % k-meams
%     labCen = zl_kmeans(cenNum, seg_vals, options);%����
    
    %�������ƶȾ���A��sim��
    disSupCen = normalize( DistanceZL(seg_vals, seg_vals, 'euclid'), 'column' );
    simSupCen = exp( -theta*disSupCen );%spnum*cenNum�ľ���
    clear disSupCen labCen
    
    % euclid�������W����ɫ�ռ䣬���е㶼��sal_lab��
    W_lab = normalize( DistanceZL(seg_vals, seg_vals, 'euclid') ); %�������W��ŷ����þ��룩
    sal_lab = exp( -theta*W_lab );%spum*spum�ľ���
    

     % euclid�������W���ߴ�ռ䣬���е㶼��sal_lab��
%     Sp_lab = normalize( DistanceZL(sp_vals, sp_vals, 'euclid') ); %�������W��ŷ����þ��룩
%     SpSal_lab = exp( -theta*Sp_lab );%spum*spum�ľ���
    
    %�ϲ�����
%     sal_lab=ColorSal_lab+ SpSal_lab;
%     sal_lab=sal_lab.*(1-C_vals);

    % Ranking
    spGraph = calSparseGraph( sal_lab, adjc, bord, spnum );%����ϡ�����
    W = sal_lab.*spGraph;%����ֻ�����������ڵĽڵ㸳ֵ(ֻ��ѡ�еļ����㸳ֵΪsal_lab)
    dd = sum(W,2);%�������D��ÿһ�н�����ͣ����Ϊ��������
    D = sparse(1:spnum,1:spnum,dd);%��Խ���(spnum*spnum)
    
    P = (D-alpha*W)\eye(spnum); %��D-alpha*W��-1��spnum*spnum��
    Sal = P*simSupCen;%��D-alpha*W��-1*SIM��spnum*cenNum��
    Sal = normalize( Sal' );%��һ��sal��cenNum*spnum��
    clear W dd D
    
   weight = calPostprocessingWeight(sal_lab,adjc,spnum);
  
    salSup = Sal.*(ones(spnum,1)*num_vals');%hij*nj(��˺ͳ˲�һ��������ĵ���Ƕ�Ӧλ�õ�ÿһ��Ԫ����ˣ���˵õ�����cenNum*spnum����)
    Isum = sum(salSup,2);%���sum��hij*nj��=Hi��������ͣ�����һ�����������õ��������г����ؿ��ÿһ��������Ӱ��ͣ�
    comSal = calDistribution(salSup,Isum,x_vals,y_vals,m,n,spnum,spnum);
    comVal= calfgDistributionSal(salSup,Isum,x_vals,y_vals, weight,m,n,spnum,spnum);%������ܶȣ��Ǹ�cenNum*1����������
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
%  %�������������Լ����Ŷ�
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
%     dd = sum(W,2);%�������D��ÿһ�н�����ͣ����Ϊ��������
%     D = sparse(1:spnum,1:spnum,dd);%��Խ���(spnum*spnum)
%     P = (D-alpha*W)\eye(spnum); %��D-alpha*W��-1��spnum*spnum��
    

     %ȡ�������½Ǳ�
    [~, index] = max(simSupCen,[],2);%ȡÿ�е����ֵ��Ҳ����ÿ�������غ��ĸ���������ƶ���󣬷��ص���ÿ�������ֵ���е�λ��(1*cenNum)
    bgIndex = unique([superpixels(1,:),superpixels(m,:),superpixels(:,1)',superpixels(:,n)']);%ȡ��������һ�б�ǵ�ֵ����ǩ��ֵ��
    bgCenInd = unique(index(bgIndex));%ȡ���߽��ϵı�����
    bgComInd = find(comVal == 0);%�ҳ����ڱ������½Ǳ�
    fgComInd = find(comVal ~= 0);%�ҳ�����ǰ�����½Ǳ�
    bgInd = unique([bgCenInd;bgComInd]);
    bgInd = setdiff( bgInd,fgComInd );%ȥ������ǰ���ĽǱ�
     
    %��ǰ��Ȩ��
%     colDistM = GetDistanceMatrix(seg_vals);
%     [clipVal, geoSigma] = EstimateDynamicParas(adjc, colDistM);
%     fgProb = EstimateBgProb( colDistM, adjc, bgIndex, clipVal, geoSigma);
    
    %Ranking
     %�������������Լ����Ŷ�
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


%%ǰ����������ϡ�����Ĺ���


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
    dd = sum(W,2);%�������D��ÿһ�н�����ͣ����Ϊ��������
    D = sparse(1:spnum,1:spnum,dd);%��Խ���(spnum*spnum)
    P = (D-alpha*W)\eye(spnum); %��D-alpha*W��-1��spnum*spnum��
    
    
%     simSupCen = simSupCen';%ת�ú�����CenNum*spnum
%     bgSal = 1-normalize(simSupCen(bgInd(1),:));%���ڱ����ľ���������ֵSik(������)
%     for i = 2:length(bgInd)
%         bgSal = bgSal.*( 1-normalize(simSupCen(bgInd(i),:)) );%���
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
    %% ---------------------- ����Աȶ� ---------------------%%  
    %��һ�ּ��㷽����ֻ�ͱ����ĳ����ؽڵ���ȣ�
%  Bg_sp_vals=[x_vals(bgInd) y_vals(bgInd)];
% con_vals=zeros(spnum,length(bgInd));
% % Sp_lab=zeros(bgInd,1);
% for i=1:spnum
%     for j=1:length(bgInd)
% %      Bg_sp_vals(j)=[x_vals(bgInd(j)) y_vals(bgInd(j))];
%        Sp_lab = normalize( DistanceZL(sp_vals(i,:), sp_vals(bgInd(j),:), 'euclid') ); %����ռ�߶��ϵ�Ȩ�أ�ŷ����þ��룩
%        Color_lab = normalize( DistanceZL(seg_vals(i,:), seg_vals(bgInd(j),:), 'euclid') ); 
%        con_vals(i,j)=exp( -theta*Sp_lab)*Color_lab;
%     end
% end
% conSal=normalize(sum(con_vals,2)); 
% conSal = normalize(P*conSal);

     %����Աȶȣ�ȫ�ֶԱȶ�)
    All_Sp_lab=normalize(DistanceZL(sp_vals, sp_vals, 'euclid'));
    All_Color_lab= normalize( DistanceZL(seg_vals,seg_vals, 'euclid'));
   All_Con_vals=All_Color_lab.*exp( -2.5*(All_Sp_lab.^2));
%     ConSal=normalize(sum(Con_vals,2));
    All_ConSal=normalize(fgSal.*(sum(All_Con_vals,2)));
%     ConSal=calContrastSal(ConSal);




     %����Աȶȣ��ڶ��ּ��㷽����ֻ�Ǻͱ����ĳ����ص����)
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
     
%   %����Աȶȣ����ھӵĳ�������ȣ�
%   Neb=zeros(spnum,spnum);
%   for i=1:spnum
%       Neb(i,:)=adjc(i,:).*(1:spnum);
%       NebIndex=find(Neb(i,:));
%    end
%       con_vals=zeros(spnum,length(NebIndex));
%    for i=1:spnum
%      for j=1:length(NebIndex)
% %       Neb_sp_vals(j)=[x_vals(NebIndex(j)) y_vals(NebIndex(j))];
%         Neb_Sp_lab = normalize( DistanceZL(sp_vals(i,:), sp_vals(NebIndex(j),:), 'euclid') ); %����ռ�߶��ϵ�Ȩ�أ�ŷ����þ��룩
%         Neb_Color_lab = normalize( DistanceZL(seg_vals(i,:), seg_vals(NebIndex(j),:), 'euclid') ); 
%         con_vals(i,:)=exp( -2.5*(Neb_Sp_lab.^2))*Neb_Color_lab.^2;
%      end
%    end
%      ConSal=normalize(sum(con_vals,2)); 
% %      ConSal=calContrastSal(ConSal);



%%--------------------------------����-------------------%%
    %�������������Լ����Ŷ�
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


%������ǰ��ϡ�����Ĺ���
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
    dd = sum(W,2);%�������D��ÿһ�н�����ͣ����Ϊ��������
    D = sparse(1:spnum,1:spnum,dd);%��Խ���(spnum*spnum)
    P = (D-alpha*W)\eye(spnum); %��D-alpha*W��-1��spnum*spnum) 
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
    %%----------------�ϲ�����---------------------%%
    %���
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



