%%
PATH1='/.../Ultimatum_Game/TF_data/';
PATH2='/.../Ultimatum_Game/ROI_chan_ICA_nnmf/chan64/';
cd(PATH1);
list_ch=[1,7,19,36,54,68,85,100,115,3,5,10,15,17,21,23,25,28,30,32,34,39,42,43,44,46,48,50,52,56,58,59,61,63,66,70,71,72,76,78,80,81,83,87,89,91,93,94,98,102,103,104,106,108,110,112,117,119,120,122,124,126,127,128];
list2=dir('proposer_*.mat');
cd /.../Ultimatum_Game/ROI_chan_ICA_nnmf
for s1=1:length(list2)
    load([PATH1,list2(s1).name]);
    DATA=DATA(list_ch,:,:);
    Data=zeros(9,200);
    for s2=1:9
        A1=reshape(DATA(s2,1:27,:),27,200);
        C1=db(abs(A1));
        data=zeros(63,27,200);
        for k=1:63
            list1=1:64;
            list1(s2)=[];
            A2=reshape(DATA(list1(k),1:27,:),27,200);
            for k1=1:27
                A3=zeros(1,200);
                for k2=1:200
                    sig1=A1(k1,k2);
                    sig2=A2(k1,k2);
                    coherresout=sig1.*conj(sig2);
                    coh1=abs(coherresout);
                    A3(k2)=coh1;
                end
                A3=(A3-mean(A3(1:7)));
                data(k,k1,:)=A3;
            end
        end
        %% %%%%%%%%%%%%%%%%
        B2=zeros(63,5,200);
        for k2=1:63
            B4=(reshape(data(k2,1:27,:),27,200));
            B4(B4<0)=0;
            [w,h] = nnmf(B4,10);
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
            B2(k2,:,:)=mappedX3(:,1:5)';
        end
        B1=reshape(B2,63*5,200);
        %%%%%%%%%%%%%%%%%%
        B4=B1;
        B4(B4<0)=0;
        [w,h] = nnmf(B4,20);
        AA=zeros(20,1);
        for t=1:20
            A2=h(t,:);
            r=std((A2(8:150)));
            AA(t)=r;%/std(A2(1:7));
        end
        h(isnan(AA),:)=[];
        AA(isnan(AA))=[];
        [H,E]=sort(AA,'descend');
        S1=h(E(1:5),:);
        S1(S1<0)=0;
        %%
        A1=abs(real(reshape(DATA(s2,:,:),42,200)));%-reshape(mean(abs(real(DATA)),1),42,200);
        B31=zeros(27,200);
        for kk1=1:27
            A2=((A1(kk1,:)));
            B31(kk1,:)=(A2-mean(A2(1:200)));
        end
        C31=zeros(5,27,200);
        for k1=1:5
            for k2=1:27
                C31(k1,k2,:)=B31(k2,:).*S1(k1,:);
            end
        end
        M=reshape(C31,27*5,200);
        [h31, mixingmatrix] = rs_nnica(M,20,0.01,500,1e-8);
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        A3=abs(imag(reshape(DATA(s2,:,:),42,200)));%-reshape(mean(abs(imag(DATA)),1),42,200);
        B32=zeros(27,200);
        for kk1=1:27
            A2=((A3(kk1,:)));
            B32(kk1,:)=(A2-mean(A2(1:200)));
        end
        C32=zeros(5,27,200);
        for k1=1:5
            for k2=1:27
                C32(k1,k2,:)=B32(k2,:).*S1(k1,:);
            end
        end
        M=reshape(C32,27*5,200);
        [h32, mixingmatrix] = rs_nnica(M,20,0.01,500,1e-8);
        %%
        X=rand(20,1);
        Y=rand(20,1);
        a1=zeros(20,1);
        a2=zeros(20,1);
        a3=zeros(20,1);
        a4=zeros(20,1);
        b1=zeros(20,1);
        b2=zeros(20,1);
        b3=zeros(20,1);
        b4=zeros(20,1);
        a=h31';
        b=h32';
        u=0.001;
        [X,Y]=GD_WY33(X,Y,a,b,a1,a2,a3,a4,b1,b2,b3,b4,u,0.001);
        EE=(a*X+b*Y);
        Data(s2,:)=EE;
    end
    save([PATH2,list2(s1).name],'Data');
end
