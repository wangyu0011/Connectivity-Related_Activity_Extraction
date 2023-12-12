%%
PATH1='./';
PATH2='./';
cd(PATH1);
list_ch=[1,7,19,36,54,68,85,100,115];
list2=dir('TF*.mat');
cd /media/wangyu/新加卷1/Ultimatum_Game/ROI_chan1
for s1=1:311%length(list2)
    load([PATH1,list2(s1).name]);
    DATA=DATA(list_ch,:,:);
    Data=zeros(9,200);
    for s2=1:9
        A1=reshape(DATA(s2,1:27,:),27,200);
        C1=(abs(A1));
        data=zeros(8,27,200);
        for k=1:8
            list1=1:9;
            list1(s2)=[];
            A2=reshape(DATA(list1(k),1:27,:),27,200);
            for k1=1:27
                A3=zeros(1,200);
                for k2=1:200
                    sig1=A1(k1,k2);
                    sig2=A2(k1,k2);
                    coherresout=sig1.*conj(sig2);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    coh1=abs(coherresout);
                    A3(k2)=coh1;
                end
                A3=(A3-mean(A3(1:7)));%/std(A3(1:7));
                data(k,k1,:)=A3;
            end
        end
        %%%%%%%%%%%%%%%%%%
        B2=zeros(8,5,200);
        for k2=1:8
%             [mappedX3, mapping]=em_pca(reshape(data(k2,:,:),42,200)',10);
%             [mappedX3,SCORE]=pca(reshape(data(k2,1:27,:),27,200));
            B4=(reshape(data(k2,1:27,:),27,200));
            B4(B4<0)=0;
            [w,h] = nnmf(B4,10);
%             [ v1] = NNPCA2014(B4);
%             h=v1'*B4;
            AA=zeros(10,1);
            for t=1:10
                A2=h(t,:);
                r=std((A2(8:150)));
                AA(t)=r/std(A2(1:7));
            end
            h(isnan(AA),:)=[];
            AA(isnan(AA))=[];
            [H,E]=sort(AA,'descend');
            mappedX3=h(E(1:5),:)';
            B2(k2,:,:)=mappedX3(:,:)';
        end
        B1=reshape(B2,5*8,200);
        %%%%%%%%%%%%%%%%%%
        B3=zeros(27,200);
        for k1=1:27
            A2=(abs(A1(k1,:)));
            B3(k1,:)=(A2-mean(A2(1:7)));%/std(A2(1:7));
        end
        t=0;
        AA1=zeros(60,200);
        for k1=1:26
            for k2=k1+1:27
                A1=B3(k1,:);
                A2=B3(k2,:);
                t=t+1;
                AA1(t,:)=A1.*A2;
            end
        end
        B3=abs(AA1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        B4=B1;
        B4(B4<0)=0;
        [w,h] = nnmf(B4,40);
        S1=h;
        data=zeros(40,1);
        for t=1:40
            A2=S1(t,:);
            r=std((A2(8:150)));
            data(t)=r/std(A2(1:7));
        end
        S1(isnan(data),:)=[];
        data(isnan(data))=[];
        [H,E]=sort(data,'descend');
        S1=S1(E(1:20),:);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        B4=abs(B3);
        B4(B4<0)=0;
        [w,h] = nnmf(B4,40);
        S2=h;
        data=zeros(40,1);
        for t=1:40
            A2=S2(t,:);
            r=std((A2(8:150)));
            data(t)=r/std(A2(1:7));
        end
        S2(isnan(data),:)=[];
        data(isnan(data))=[];
        [H,E]=sort(data,'descend');
        S2=S2(E(1:20),:);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [A,B,R,U,V] = canoncorr(S1',S2');
        U1 =S1'*A;
        V1 =S2'*B;
        P=U1.*V1;
        R(R<0.5)=0;
        P1=P*R';
        Data(s2,:)=P1;
    end
    save([PATH2,'chan9_',list2(s1).name],'Data');
end





















