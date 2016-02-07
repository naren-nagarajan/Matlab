clc
clear
load train_sp2015_v14
train = train_sp2015_v14;

% SVM

labels=zeros(10000,1);
labels(1:5000,1)=1;
labels(5001:10000,1)=3;

features=zeros(10000,4);
features(1:5000,:)=train(1:5000,:);
features(5001:10000,:)=-train(10001:15000,:);

min_features=min(features);
max_features=max(features);

x_bar=zeros(10000,4);
for i=1:10000
    x_bar(i,:)=(features(i,:)-min_features)./(max_features-min_features);
end

A=sparse(x_bar);
libsvmwrite('svmdata.txt',labels,A);

[svm_label, svm_matrix]=libsvmread('svmdata.txt');
% bestcv = 0;
% for log2c = -1:3,
%   for log2g = -4:1,
%     cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
%     cv = svmtrain(svm_label,svm_matrix, cmd);
%     if (cv >= bestcv),
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end

model=svmtrain(svm_label,svm_matrix,'-t 2 -c 8 -g 2 -b 0');
[final_labels]=svmpredict(svm_label, svm_matrix, model);

% 0: nSV = 2935, Accuracy = 88.28%
% 2: nSV = 2279, Accuracy = 91.16%
 
%%%%%%%%%%%%%%
%%% Part 2 
%%% C Means

cov1=cov(train);
%%%%%% Clusters = 3

c1_temp=zeros(1,15000);
c1=ones(1,15000);
mu_1=train(2343,:);
mu_2=train(5656,:);
mu_3=train(8787,:);
distance=zeros(3,15000);
counter=0;
flag=1;
while(norm(c1-c1_temp)~=0)
    c1_temp=c1;
    c1=ones(1,15000);
    for i=1:15000
     % For Mahalanobis Distances
       distance(1,i)=(train(i,:)-mu_1)*inv(cov1)*(train(i,:)-mu_1)';
       distance(2,i)=(train(i,:)-mu_2)*inv(cov1)*(train(i,:)-mu_2)';
       distance(3,i)=(train(i,:)-mu_3)*inv(cov1)*(train(i,:)-mu_3)';
%             distance(1,i)=norm(train(i,1:4)-mu_1);
%             distance(2,i)=norm(train(i,1:4)-mu_2);
%             distance(3,i)=norm(train(i,1:4)-mu_3);
        if((distance(1,i)<distance(2,i))&&(distance(1,i)<distance(3,i)))
            c1(i)=1;
        else
            if((distance(2,i)<distance(1,i))&&(distance(2,i)<distance(3,i)))
                c1(i)=2;
            else
                if((distance(3,i)<distance(1,i))&&(distance(3,i)<distance(2,i)))
                c1(i)=3;
                end
            end
        end
    end
    
    a=1;
    b=1;
    c=1;
    clear cluster1;
    clear cluster2;
    clear cluster3;
    for j=1:15000
        if(c1(j)==1)
            cluster1(a,1:4)=train(j,1:4);
            a=a+1;
        end
        if(c1(j)==2)
            cluster2(b,1:4)=train(j,1:4);
            b=b+1;
        end
        if(c1(j)==3)
            cluster3(c,1:4)=train(j,1:4);
            c=c+1;
        end
    end
    mu_1=mean(cluster1(:,1:4));
    mu_2=mean(cluster2(:,1:4));
    mu_3=mean(cluster3(:,1:4));
    counter=counter+1;
end
    c3mean1=mu_1;
    c3mean2=mu_2;
    c3mean3=mu_3;
    c3cluster1=cluster1;
    c3cluster2=cluster2;
    c3cluster3=cluster3;
    
%%%% Clusters = 4

c4_temp=zeros(1,15000);
c4=ones(1,15000);
mu_1=train(2343,:);
mu_2=train(5656,:);
mu_3=train(8787,:);
mu_4=train(12121,:);
distance=zeros(3,15000);
counter=0;
while(norm(c4-c4_temp)~=0)
    c4_temp=c4;
    c4=ones(1,15000);
    for i=1:15000
        % For Mahalanobis Distances
          distance(1,i)=(train(i,:)-mu_1)*inv(cov1)*(train(i,:)-mu_1)';
          distance(2,i)=(train(i,:)-mu_2)*inv(cov1)*(train(i,:)-mu_2)';
          distance(3,i)=(train(i,:)-mu_3)*inv(cov1)*(train(i,:)-mu_3)';
          distance(4,i)=(train(i,:)-mu_4)*inv(cov1)*(train(i,:)-mu_4)';
%             distance(1,i)=norm(train(i,1:4)-mu_1);
%             distance(2,i)=norm(train(i,1:4)-mu_2);
%             distance(3,i)=norm(train(i,1:4)-mu_3);
%             distance(4,i)=norm(train(i,1:4)-mu_4);
        if((distance(1,i)<distance(2,i))&&(distance(1,i)<distance(3,i))&&(distance(1,i)<distance(4,i)))
            c4(i)=1;
        else
            if((distance(2,i)<distance(1,i))&&(distance(2,i)<distance(3,i))&&(distance(2,i)<distance(4,i)))
                c4(i)=2;
            else
                if((distance(3,i)<distance(1,i))&&(distance(3,i)<distance(2,i))&&(distance(3,i)<distance(4,i)))
                c4(i)=3;
            else
                if((distance(4,i)<distance(1,i))&&(distance(4,i)<distance(2,i))&&(distance(4,i)<distance(3,i)))
                c4(i)=4;
                end
                end
            end
        end
    end
    
    a=1;
    b=1;
    c=1;
    d=1;
    clear cluster1;
    clear cluster2;
    clear cluster3;
    clear cluster4;
    for j=1:15000
        if(c4(j)==1)
            cluster1(a,1:4)=train(j,1:4);
            a=a+1;
        end
        if(c4(j)==2)
            cluster2(b,1:4)=train(j,1:4);
            b=b+1;
        end
   
        if(c4(j)==3)
            cluster3(c,1:4)=train(j,1:4);
            c=c+1;
        end
        if(c4(j)==4)
            cluster4(d,1:4)=train(j,1:4);
            d=d+1;
        end
    end
    mu_1=mean(cluster1(:,:));
    mu_2=mean(cluster2(:,:));
    mu_3=mean(cluster3(:,:));
    mu_4=mean(cluster4(:,:));
    counter=counter+1;
end
    c4mean1=mu_1;
    c4mean2=mu_2;
    c4mean3=mu_3;
    c4mean4=mu_4;
    c4cluster1=cluster1;
    c4cluster2=cluster2;
    c4cluster3=cluster3;
    c4cluster4=cluster4;

%%%%% Clusters = 5

c5_temp=zeros(1,15000);
c5=ones(1,15000);
mu_1=train(2343,:);
mu_2=train(5656,:);
mu_3=train(8787,:);
mu_4=train(12121,:);
mu_5=train(14141,:);
distance=zeros(5,15000);
counter=0;
while(norm(c5-c5_temp)~=0)
    c5_temp=c5;
    c5=ones(1,15000);
    for i=1:15000
        % For Mahalanobis Distances
          distance(1,i)=(train(i,:)-mu_1)*inv(cov1)*(train(i,:)-mu_1)';
          distance(2,i)=(train(i,:)-mu_2)*inv(cov1)*(train(i,:)-mu_2)';
          distance(3,i)=(train(i,:)-mu_3)*inv(cov1)*(train(i,:)-mu_3)';
          distance(4,i)=(train(i,:)-mu_4)*inv(cov1)*(train(i,:)-mu_4)';
          distance(5,i)=(train(i,:)-mu_5)*inv(cov1)*(train(i,:)-mu_5)';
%             distance(1,i)=norm(train(i,1:4)-mu_1);
%             distance(2,i)=norm(train(i,1:4)-mu_2);
%             distance(3,i)=norm(train(i,1:4)-mu_3);
%             distance(4,i)=norm(train(i,1:4)-mu_4);
%             distance(5,i)=norm(train(i,1:4)-mu_5);
        if((distance(1,i)<distance(2,i))&&(distance(1,i)<distance(3,i))&&(distance(1,i)<distance(4,i))&&(distance(1,i)<distance(5,i)))
            c5(i)=1;
        else
            if((distance(2,i)<distance(1,i))&&(distance(2,i)<distance(3,i))&&(distance(2,i)<distance(4,i))&&(distance(2,i)<distance(5,i)))
                c5(i)=2;
            else
                if((distance(3,i)<distance(1,i))&&(distance(3,i)<distance(2,i))&&(distance(3,i)<distance(4,i))&&(distance(3,i)<distance(5,i)))
                c5(i)=3;
            else
                if((distance(4,i)<distance(1,i))&&(distance(4,i)<distance(2,i))&&(distance(4,i)<distance(3,i))&&(distance(4,i)<distance(5,i)))
                c5(i)=4;
            else
                if((distance(5,i)<distance(1,i))&&(distance(5,i)<distance(2,i))&&(distance(5,i)<distance(3,i))&&(distance(5,i)<distance(4,i)))
                c5(i)=5;    
                end
                end
                end
            end
        end
    end
    
    a=1;
    b=1;
    c=1;
    d=1;
    e=1;
    clear cluster1;
    clear cluster2;
    clear cluster3;
    clear cluster4;
    clear cluster5;
    for j=1:15000
        if(c5(j)==1)
            cluster1(a,1:4)=train(j,1:4);
            a=a+1;
        end
        if(c5(j)==2)
            cluster2(b,1:4)=train(j,1:4);
            b=b+1;
        end
   
        if(c5(j)==3)
            cluster3(c,1:4)=train(j,1:4);
            c=c+1;
        end
        if(c5(j)==4)
            cluster4(d,1:4)=train(j,1:4);
            d=d+1;
        end
        if(c5(j)==5)
            cluster5(e,1:4)=train(j,1:4);
            e=e+1;
        end
    end
    mu_1=mean(cluster1(:,:));
    mu_2=mean(cluster2(:,:));
    mu_3=mean(cluster3(:,:));
    mu_4=mean(cluster4(:,:));
    mu_5=mean(cluster5(:,:));
    counter=counter+1;
end
    c5mean1=mu_1;
    c5mean2=mu_2;
    c5mean3=mu_3;
    c5mean4=mu_4;
    c5mean5=mu_5;
    c5cluster1=cluster1;
    c5cluster2=cluster2;
    c5cluster3=cluster3;
    c5cluster4=cluster4;
    c5cluster5=cluster5;


% %%%%%%%%%%%%%%%%%%%%%%%%

%Part 3

% ANN
clc
clear
load train_sp2015_v14
train = train_sp2015_v14;

class1=train(1:5000,:);
class2=train(10001:15000,:);
train11=[class1;class2];
minvec=min(train11);
maxvec=max(train11);
for i=1:10000
    train11(i,:)=(train11(i,:)-minvec)./(maxvec-minvec);
end
j=1;
for i=1:2:10000
    train1(i,:)=train11(j,:);
    train1(i+1,:)=train11(j+5000,:);
    j=j+1;
end
weight1 = linspace(0.01,0.02,36);
weight2=linspace(0.02,0.03,9);
bias=0.5;
alpha=0.3;

for s=1:50
    for i=1:10000
        if mod(i,2)==0
            target=0;
        else target=1;
        end     
        k=1;
        for j=1:9
                hidden(i,j)=train1(i,:)*weight1(k:k+3)'+bias;
                hidden_out(i,j)=1/(1+exp(-hidden(i,j)));
                k=k+4;
        end
        
        netout(i)=hidden_out(i,:)*weight2'+bias;
        output(i)=1/(1+exp(-netout(i)));
        error(i)=target-output(i);
        delta2(i)=output(i)*(1-output(i))*error(i);
        for j=1:9
            weightchange2(i,j)=delta2(i)*hidden_out(i,j)*alpha;
            delta1(i,j)=delta2(i)*hidden_out(i,j)*(1-hidden_out(i,j))*weight1(j);
        end
        k=1;
        for j=1:9
            for p=1:4
                weightchange1(i,k)=delta1(i,j)*train1(i,p)*alpha;
                k=k+1;
            end
        end
        weight1=weight1+weightchange1(i,1:36);
        weight2=weight2+weightchange2(i,1:9);
    end
    TSS=norm(error);
end

for i=1:10000
    k=1;
    for j=1:9
        hiddenfinal(i,j)=train11(i,:)*weight1(k:k+3)'+bias;
        hiddenoutfinal(i,j)=1/(1+exp(-hiddenfinal(i,j)));
        k=k+4;
    end
        
    netoutfinal(i)=hiddenoutfinal(i,:)*weight2'+bias;
    outputfinal(i)=1/(1+exp(-netoutfinal(i)));
end

for i=1:10000
    if outputfinal(i)>0.5
        outputfinal(i)=1;
    else outputfinal(i)=0;
    end
end
check1=0;
check2=0;
for i=1:5000
    if outputfinal(i)==0
        check1=check1+1;
    end
end
for i=5001:10000
    if outputfinal(i)==1
        check2=check2+1;
    end
end
wrong=check1+check2;
error_final=(wrong*100)/10000;


%%%%%%%%%%%%%%%%%%%%

%%%% Part 4

load test_sp2015_v14
train1=train_sp2015_v14;
test=test_sp2015_v14;

class1=train1(1:5000,1:4);
class2=train1(5001:10000,1:4);
class3=train1(10001:15000,1:4);
train11=[class1;class2;class3];
min1=min(train11);
max1=max(train11);

i=1;
for j=1:3:15000
train(j,:)=train11(i,:);
train(j+1,:)=train11(i+5000,:);
train(j+2,:)=train11(i+10000,:);
i=i+1;
end

weight1=linspace(0.01,0.02,36);
weight2=linspace(0.02,0.03,27);
bias=0.5;
alpha=1;
X=[1 0 0;0 1 0;0 0 1];
target=repmat(X,[5000,1]);

s=1;
delta1=zeros(15000,9);
for s=1:5
    for i=1:15000
        k=1;
        for j=1:9
            hidden(i,j)=train(i,:)*weight1(k:k+3)'+bias;
            k=k+4;
            hidden_out(i,j)=1/(1+exp(-hidden(i,j)));
        end
        k=1;
        for j=1:3
            netout(i,j)=hidden_out(i,:)*weight2(k:k+8)'+bias;
            k=k+9;
            output(i,j)=1/(1+exp(-netout(i,j)));
            delta2(i,j)=(output(i,j)*(1-output(i,j)))*(target(i,j)-output(i,j));
            error(i,j)=(target(i,j)-output(i,j));
        end
        k=1;
        for j=1:3
            for p=1:9
                weightchange2(i,k)=delta2(i,j)*hidden_out(i,p)*alpha;
                k=k+1;
            end
        end
        k=1;
        for j=1:9
            for p=1:3
                delta1(i,j)=delta1(i,j)+(hidden_out(i,j)*(1-hidden_out(i,j))*delta2(i,p)*weight2(k));
                k=k+1;
            end
        end
        k=1;
        for j=1:9
            for p=1:4
                weightchange1(i,k)=delta1(i,j)*train(i,p)*alpha;
                k=k+1;
            end
        end
        weight2=weight2+weightchange2(i,1:27);
        weight1=weight1+weightchange1(i,1:36);
    end
    TSS=norm(error);
end

for i=1:15000
    k=1;
    for j=1:9
        hidden(i,j)=test(i,:)*weight1(k:k+3)'+bias;
        hidden_out(i,j)=1/(1+exp(-hidden(i,j)));
        k=k+4;
    end        
    k=1;
    for j=1:3
        netout(i,j)=hidden_out(i,:)*weight2(k:k+8)'+bias;
        output(i,j)=1/(1+exp(-netout(i,j)));
        k=k+9;
    end
end
final=ones(1,15000);
for j=1:15000
    if max(output(j,:))==output(j,1)
        final(j)=1;
    elseif max(output(j,:))==output(j,2)
        final(j)=2;
    elseif  max(output(j,:))==output(j,3)
        final(j)=3;
    end
end

confusion=zeros(3,3);
reality=[2,3,1,3,1,2];
for i=1:6:15000
    for j=1:6
        confusion(reality(j),final(1,i+j-1))=confusion(reality(j),final(1,i+j-1))+1;
    end
end
correct=0;
for i=1:3
    correct=correct+confusion(i,i);
end
efficiency=correct/15000*100;
error=100-efficiency;