clc
clear
load narenn-classified-q1pt1.txt
estimated_test = narenn_classified_q1pt1;

% Test data Confusion Matrix

confusion=zeros(3,3);
reality=[2,3,1,3,1,2];
for i=1:6:15000
    for j=1:6
    confusion(reality(j),estimated_test(i+j-1,1))=confusion(reality(j),estimated_test(i+j-1,1))+1;
    end
end
confusion;
correct=0;
for i=1:3
    correct=correct+confusion(i,i);
end
efficiency_1=correct/15000*100;
error_1=100-efficiency_1;

%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%

load train_sp2015_v14;
train = train_sp2015_v14;

load test_sp2015_v14; 
test = test_sp2015_v14; 

%zero mean
mean=mean(train);
zero_mean=zeros(15000,4);
for i=1:15000
    zero_mean(i,:)=train(i,:)-mean;
end
x_minus_mu=zeros(15000,4);
for i=1:15000       
        x_minus_mu(i,:)=zero_mean(i,:)-mean;    
end

%Covariance and aproiri probability of each class

covariance=x_minus_mu(1:15000,:)'*x_minus_mu(1:15000,:)./15000;

[eigenval,eigenvec]=eig(covariance);
new_training=train*eigenvec(:,3:4);
new_test=test*eigenvec(:,3:4);

%%%%%%%%%%%%%%%%%%% Training the classifier %%%%%%%%%%%%%%%%%%% 


% Sum of all feature vectors for each class
sum1=zeros(3,2);
k=1;
for i=1:15000
    for j=1:2
        sum1(k,j)=sum1(k,j)+new_training(i,j);    
    end
    if(mod(i,5000)==0)
            k=k+1;
    end
    if(i==15000)
        k=3;
    end
end

% Mean and (X-Mean) for each class
mean=ones(3,2);
for i=1:3
    for j=1:2
        mean(i,j)=sum1(i,j)/5000;
    end
end
x_minus_mu=zeros(15000,2);
for i=1:5000        % Class 1
    for j=1:2
        x_minus_mu(i,j)=new_training(i,j)-mean(1,j);    
    end
end
for i=5001:10000    % Class 2
    for j=1:2
        x_minus_mu(i,j)=new_training(i,j)-mean(2,j);    
    end
end        
for i=10001:15000   % Class 3
    for j=1:2
        x_minus_mu(i,j)=new_training(i,j)-mean(3,j);    
    end
end

%Covariance and aproiri probability of each class
covariance1=x_minus_mu(1:5000,:)'*x_minus_mu(1:5000,:)./5000;
covariance2=x_minus_mu(5001:10000,:)'*x_minus_mu(5001:10000,:)./5000;
covariance3=x_minus_mu(10001:15000,:)'*x_minus_mu(10001:15000,:)./5000;
p_c1=1/3;
p_c2=1/3;
p_c3=1/3;
mu_1=mean(1,:);
mu_2=mean(2,:);
mu_3=mean(3,:);

%%%%% Training Data

estimate=zeros(15000,1);% Classifier estimate
count_equal=0;% Number of times we get equal discriminant functions
d=2;
for i=1:15000
    x=new_training(i,:);
    %Discriminant function for each class
    g1= -d/2*log(2*pi)+log(p_c1)-(0.5*log(det(covariance1)))-(0.5*(x-mu_1)*(covariance1)^(-1)*(x-mu_1)');
    g2= -d/2*log(2*pi)+log(p_c2)-(0.5*log(det(covariance2)))-(0.5*(x-mu_2)*(covariance2)^(-1)*(x-mu_2)');
    g3= -d/2*log(2*pi)+log(p_c3)-(0.5*log(det(covariance3)))-(0.5*(x-mu_3)*(covariance3)^(-1)*(x-mu_3)');
    %classification depending on valu5e of discriminant function
    if(g1>g2)           %If class 1 is greater than class 2
        if(g1>g3)
            estimate(i,1)=1;
        elseif(g3>g1)
            estimate(i,1)=3;
        else
                a = 0;
                b = 1;
                r = round((b-a).*rand(1,1) + a);
                if(r==0)
                    estimate(i,1)=3;
                else
                    estimate(i,1)=1;
                end
                count_equal=count_equal+1;        
        end
    elseif(g2>g1)       %If class 2 is greater than class 1
        if(g2>g3)
            estimate(i,1)=2;
        elseif(g3>g2)
                estimate(i,1)=3;
        else
            a = 0;
            b = 1;
            r = round((b-a).*rand(1,1) + a);
            if(r==0)
                estimate(i,1)=2;
            else
                estimate(i,1)=3;
            end
            count_equal=count_equal+1;
        end
    elseif(g1==g2)      %if class 1 is equal to class 2
        count_equal=count_equal+1;
        a = 0;
        b = 1;
        r = round((b-a).*rand(1,1) + a);
        if(r==0)
            estimate(i,1)=1;
        else
            estimate(i,1)=2;
        end
    else                %if still not estimated
        a = 0;
        b = 2;
        r = round((b-a).*rand(1,1) + a);
        if(r==0)
            estimate(i,1)=1;
        elseif(r==1)
            estimate(i,1)=2;
        else
            estimate(i,1)=3;
        end
    end
end

%Confusion Matrix - Training Data
confusion=zeros(3,3);
for i=1:5000        %Class 1
    if(estimate(i,1)==1)
        confusion(1,1)=confusion(1,1)+1;
    end
    if(estimate(i,1)==2)
        confusion(1,2)=confusion(1,2)+1;
    end
    if(estimate(i,1)==3)
        confusion(1,3)=confusion(1,3)+1;
    end    
end
for i=5001:10000    %Class 2
    if(estimate(i,1)==1)
        confusion(2,1)=confusion(2,1)+1;
    end
    if(estimate(i,1)==2)
        confusion(2,2)=confusion(2,2)+1;
    end
    if(estimate(i,1)==3)
        confusion(2,3)=confusion(2,3)+1;
    end    
end
for i=10001:15000   %Class 3
    if(estimate(i,1)==1)
        confusion(3,1)=confusion(3,1)+1;
    end
    if(estimate(i,1)==2)
        confusion(3,2)=confusion(3,2)+1;
    end
    if(estimate(i,1)==3)
        confusion(3,3)=confusion(3,3)+1;
    end    
end

%Efficiency and error from confusion matrix
sum_diagonal_training=0;
for i=1:3
    sum_diagonal_training=sum_diagonal_training+confusion(i,i); %calculate sum of diagonal elements
                                                                %of confusion matrix 
end
efficiency_training=sum_diagonal_training*100/15000;  %efficiency
error_training=100-efficiency_training;               %error 

%%%%% Test Data


estimate_test=zeros(15000,1); % Classifier estimate
count_equal_test=0; % Number of times we get equal discriminant functions
for i=1:15000
    x=new_test(i,:);
    % Disriminant functions for each class
    g1_test=-(0.5*log(det(covariance1)))-(0.5*(x-mu_1)*(covariance1)^(-1)*(x-mu_1)');%+log(p_c1);
    g2_test=-(0.5*log(det(covariance2)))-(0.5*(x-mu_2)*(covariance2)^(-1)*(x-mu_2)');%+log(p_c2);
    g3_test=-(0.5*log(det(covariance3)))-(0.5*(x-mu_3)*(covariance3)^(-1)*(x-mu_3)');%+log(p_c3);
    % Classification depending on value of discriminant function
    if(g1_test>g2_test)         %Class 1 greater than class 2
        if(g1_test>g3_test)
            estimate_test(i,1)=1;
        elseif(g3_test>g1_test)
            estimate_test(i,1)=3;
        else
                a = 0;
                b = 1;
                r = round((b-a).*rand(1,1) + a);
                if(r==0)
                    estimate_test(i,1)=3;
                else
                    estimate_test(i,1)=1;
                end
                count_equal_test=count_equal_test+1;        
        end
    elseif(g2_test>g1_test)     %Class 2 greater than class 1
        if(g2_test>g3_test)
            estimate_test(i,1)=2;
        elseif(g3_test>g2_test)
                estimate_test(i,1)=3;
        else
            a = 0;
            b = 1;
            r = round((b-a).*rand(1,1) + a);
            if(r==0)
                estimate_test(i,1)=2;
            else
                estimate_test(i,1)=3;
            end
            count_equal_test=count_equal_test+1;
        end
    elseif(g1_test==g2_test)    %Class 1 equal to class 2
        count_equal_test=count_equal_test+1;
        a = 0;
        b = 1;
        r = round((b-a).*rand(1,1) + a);
        if(r==0)
            estimate_test(i,1)=1;
        else
            estimate_test(i,1)=2;
        end
    else                        %  If still not estimated
        a = 0;
        b = 2;
        r = round((b-a).*rand(1,1) + a);
        if(r==0)
            estimate_test(i,1)=1;
        elseif(r==1)
            estimate_test(i,1)=2;
        else
            estimate_test(i,1)=3;
        end
    end
end

%Counting number of feature vecotors sorted in each class
class_count(4,1)=0;
for i=1:15000
    if (estimate_test(i,1)==1)
        class_count(1,1)=class_count(1,1)+1;    %Class 1
    elseif (estimate_test(i,1)==2)
        class_count(2,1)=class_count(2,1)+1;    %Class 2
    elseif (estimate_test(i,1)==3)  
        class_count(3,1)=class_count(3,1)+1;    %Class 3
    else 
        class_count(4,1)=class_count(4,1)+1;    %not sorted into a class
    end
end

%pca_confusion matrix - Test Data

pca_confusion=zeros(3,3);
reality=[2,3,1,3,1,2];
for i=1:6:15000
    for j=1:6
    pca_confusion(reality(j),estimate_test(i+j-1,1))=pca_confusion(reality(j),estimate_test(i+j-1,1))+1;
    end
end

correct=0;
for i=1:3
    correct=correct+pca_confusion(i,i);
end
pca_efficiency=correct/15000*100;
pca_error=100-pca_efficiency

a=fopen('narenn-classified-q2pt1-PCA-test.txt','w');
b=fopen('narenn-classified-q2pt1-PCA-train.txt','w');
for i=1:15000
    fprintf(a,'%d \n',estimate_test(i));
    fprintf(b,'%d \n',estimate(i));
end
fclose(a);
fclose(b);

%%%%%%%%%%%%% KNNR %%%%%%%%%%%%%%%%%%

load train_sp2015_v14;
knnr_train = train_sp2015_v14;

load test_sp2015_v14; 
knnr_test = test_sp2015_v14; 

for K=1:2:5
    for m=1:15000
        distance=sum((repmat(knnr_test(m,:),15000,1) - knnr_train).^2,2);
        [sorted_distance index]=sort(distance,'ascend');
        minimum=index(1:K);  
        for i=1:K
            if minimum(i)<=5000
                class(i)=1;
            elseif minimum(i)>5000 && minimum(i)<=10000
                class(i)=2;
            else
                class(i)=3;
            end
        end
        knnr_estimate(m,1)=mode(class);
    end
knnr_confusion=zeros(3,3);
reality=[2,3,1,3,1,2];
for i=1:6:15000
    for j=1:6
    knnr_confusion(reality(j),knnr_estimate(i+j-1,1))=knnr_confusion(reality(j),knnr_estimate(i+j-1,1))+1;
    end
end
correct=0;
for i=1:3
    correct=correct+knnr_confusion(i,i);
end
knnr_efficiency=correct/15000*100;
knnr_error=100-knnr_efficiency

if(K==1)
    a=fopen('narenn-classified-q2pt1-knnr-k=1.txt','w');
    for i=1:15000
    fprintf(a,'%d \n',knnr_estimate(i));
    end
    fclose(a);
end
if(K==3)
    a=fopen('narenn-classified-q2pt1-knnr-k=3.txt','w');
    for i=1:15000
    fprintf(a,'%d \n',knnr_estimate(i));
    end
    fclose(a);
end
if(K==5)
    a=fopen('narenn-classified-q2pt1-knnr-k=5.txt','w');
    for i=1:15000
    fprintf(a,'%d \n',knnr_estimate(i));
    end
    fclose(a);
end
end

%%%%%%%%%%%%%%%%%%%% Ho-Kashyap %%%%%%%%%%%%%%%%%%%%%


load train_sp2015_v14;
train = train_sp2015_v14;

load test_sp2015_v14; 
test = test_sp2015_v14; 
x=ones(5000,1);
x1=ones(10000,1);

% hyperplane between 1-2_3

Y_12=[x train(1:5000,:);-x -train(5001:10000,:);-x -train(10001:15000,:)];
a_12=zeros(5,1);
b=ones(15000,1);
learning_rate=0.9;
iterations_12=0;
a_12_new=ones(5,1);
while(norm(a_12_new-a_12)>0.0001)
    a_12=a_12_new;
    error_1=Y_12*a_12-b;
    b=b+(learning_rate*(error_1+abs(error_1)));
    a_12_new=(Y_12'*Y_12)^-(1)*Y_12'*b;
    iterations_12=iterations_12+1;
end

% hyperplane between 2-3_1

Y_23=[x train(5001:10000,:);-x -train(10001:15000,:);-x -train(1:5000,:)];
a_23=zeros(5,1);
b=ones(15000,1);
learning_rate=0.9;
iterations_23=0;
a_23_new=ones(5,1);
while(norm(a_23_new-a_23)>0.0001)
    a_23=a_23_new;
    error_2=Y_23*a_23-b;
    b=b+(learning_rate*(error_2+abs(error_2)));
    a_23_new=(Y_23'*Y_23)^-(1)*Y_23'*b;
end

% hyperplane between 3-1_2

Y_31=[x train(10001:15000,:);-x -train(1:5000,:);-x -train(5001:10000,:)];
a_31=zeros(5,1);
b=ones(15000,1);
learning_rate=0.9;
flag=0;
iterations_31=0;
a_31_new=ones(5,1);
while(norm(a_31_new-a_31)>0.0001)
     a_31=a_31_new;
    error_3=Y_31*a_31-b;
    b=b+(learning_rate*(error_3+abs(error_3)));
    a_31_new=(Y_31'*Y_31)^-(1)*Y_31'*b;
end

%%%%%%%%%%Training data Classification

hyperplane_train_estimate=zeros(15000,3);
hyperplane_train_estimate(:,1)=train*a_12_new(2:5,1);
hyperplane_train_estimate(:,2)=train*a_23_new(2:5,1);
hyperplane_train_estimate(:,3)=train*a_31_new(2:5,1);
hyperplane_train_final=ones(15000,1);
hyperplane_train_temp=hyperplane_train_final;
for i=1:15000
    if(hyperplane_train_estimate(i,1)>hyperplane_train_estimate(i,2))
        if(hyperplane_train_estimate(i,1)>hyperplane_train_estimate(i,3))
            hyperplane_train_final(i,1)=1;
        else 
            hyperplane_train_final(i,1)=3;
        end
    else
        if(hyperplane_train_estimate(i,3)>hyperplane_train_estimate(i,2))
            hyperplane_train_final(i,1)=3;
        else 
            hyperplane_train_final(i,1)=2;
        end
    end
end

%%%%%%% Training data confusion matrix

hyperplane_train_confusion=zeros(3,3);
for i=1:5000        %Class 1
    if(hyperplane_train_final(i,1)==1)
        hyperplane_train_confusion(1,1)=hyperplane_train_confusion(1,1)+1;
    end
    if(hyperplane_train_final(i,1)==2)
        hyperplane_train_confusion(1,2)=hyperplane_train_confusion(1,2)+1;
    end
    if(hyperplane_train_final(i,1)==3)
        hyperplane_train_confusion(1,3)=hyperplane_train_confusion(1,3)+1;
    end    
end
for i=5001:10000    %Class 2
    if(hyperplane_train_final(i,1)==1)
        hyperplane_train_confusion(2,1)=hyperplane_train_confusion(2,1)+1;
    end
    if(hyperplane_train_final(i,1)==2)
        hyperplane_train_confusion(2,2)=hyperplane_train_confusion(2,2)+1;
    end
    if(hyperplane_train_final(i,1)==3)
        hyperplane_train_confusion(2,3)=hyperplane_train_confusion(2,3)+1;
    end    
end
for i=10001:15000   %Class 3
    if(hyperplane_train_final(i,1)==1)
        hyperplane_train_confusion(3,1)=hyperplane_train_confusion(3,1)+1;
    end
    if(hyperplane_train_final(i,1)==2)
        hyperplane_train_confusion(3,2)=hyperplane_train_confusion(3,2)+1;
    end
    if(hyperplane_train_final(i,1)==3)
        hyperplane_train_confusion(3,3)=hyperplane_train_confusion(3,3)+1;
    end    
end
correct=0;
for i=1:3
    correct=correct+hyperplane_train_confusion(i,i);
end
confusion_training_efficiency=correct/15000*100;
confusion_training_error=100-confusion_training_efficiency;


%%%%%% Test Data Classification

hyperplane_estimate=zeros(15000,3);
hyperplane_estimate(:,1)=test*a_12_new(2:5,1);
hyperplane_estimate(:,2)=test*a_23_new(2:5,1);
hyperplane_estimate(:,3)=test*a_31_new(2:5,1);
hyperplane_final=ones(15000,1);
hyperplane_test_temp=hyperplane_final;
for i=1:15000
    if(hyperplane_estimate(i,1)>hyperplane_estimate(i,2))
        if(hyperplane_estimate(i,1)>hyperplane_estimate(i,3))
            hyperplane_final(i,1)=1;
        else 
            hyperplane_final(i,1)=3;
        end
    else
        if(hyperplane_estimate(i,3)>hyperplane_estimate(i,2))
            hyperplane_final(i,1)=3;
        else 
            hyperplane_final(i,1)=2;
        end
    end
end

%%%%% Test data confusion matrix

hyperplane_confusion=zeros(3,3);
reality=[2,3,1,3,1,2];
for i=1:6:15000
    for j=1:6
    hyperplane_confusion(reality(j),hyperplane_final(i+j-1,1))=hyperplane_confusion(reality(j),hyperplane_final(i+j-1,1))+1;
    end
end
correct=0;
for i=1:3
    correct=correct+hyperplane_confusion(i,i);
end
confusion_efficiency=correct/15000*100;
confusion_error=100-confusion_efficiency;
hyperplane_confusion;

a=fopen('narenn-classified-q2pt1-ho-kashyap-test.txt','w');
b=fopen('narenn-classified-q2pt1-ho-kashyap-train.txt','w');
for i=1:15000
    fprintf(a,'%d \n',hyperplane_final(i));
    fprintf(b,'%d \n',hyperplane_train_final(i));
end
fclose(a);
fclose(b);