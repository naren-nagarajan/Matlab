clc
clear

load train_sp2015_v14;
train = train_sp2015_v14;

load test_sp2015_v14; 
test = test_sp2015_v14; 

%check for Gaussian Distribution
%hist(train);

%%%%%%%%%%%%%%%%%%% Training the classifier %%%%%%%%%%%%%%%%%%% 

% Sum of all feature vectors for each class
sum=zeros(3,4);
k=1;
for i=1:15000
    for j=1:4
        sum(k,j)=sum(k,j)+train(i,j);    
    end
    if(mod(i,5000)==0)
            k=k+1;
    end
    if(i==15000)
        k=3;
    end
end

% Mean and (X-Mean) for each class
mean=ones(3,4);
for i=1:3
    for j=1:4
        mean(i,j)=sum(i,j)/5000;
    end
end
x_minus_mu=zeros(15000,4);
for i=1:5000        % Class 1
    for j=1:4
        x_minus_mu(i,j)=train(i,j)-mean(1,j);    
    end
end
for i=5001:10000    % Class 2
    for j=1:4
        x_minus_mu(i,j)=train(i,j)-mean(2,j);    
    end
end        
for i=10001:15000   % Class 3
    for j=1:4
        x_minus_mu(i,j)=train(i,j)-mean(3,j);    
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

%%%%%%%%%%%%%%%%%%% Training Data %%%%%%%%%%%%%%%%%%% 

estimate=zeros(15000,1);% Classifier estimate
count_equal=0;% Number of times we get equal discriminant functions
for i=1:15000
    x=train(i,:);
    %Discriminant function for each class
    g1= 4/2*log(2*pi)-log(p_c1)-(0.5*log(det(covariance1)))-(0.5*(x-mu_1)*(covariance1)^(-1)*(x-mu_1)');
    g2=4/2*log(2*pi)-log(p_c2)-(0.5*log(det(covariance2)))-(0.5*(x-mu_2)*(covariance2)^(-1)*(x-mu_2)');
    g3=4/2*log(2*pi)-log(p_c3)-(0.5*log(det(covariance3)))-(0.5*(x-mu_3)*(covariance3)^(-1)*(x-mu_3)');
    %classification depending on value of discriminant function
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

%Confusion Matrix
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
H=confusion
P=error_training

%%%%%%%%%%%%%%%%%%% Test Data %%%%%%%%%%%%%%%%%%%

estimate_test=zeros(15000,1); % Classifier estimate
count_equal_test=0; % Number of times we get equal discriminant functions
for i=1:15000
    x=test(i,:);
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

%saving test data in an ASCII file
a=fopen('narenn-classified-q1pt1.txt','w');
for i=1:15000
    fprintf(a,'%d \n',estimate_test(i));
end
fclose(a);

%saving training data in an ASCII file
b=fopen('narenn-classified-q1pt1-training_data.txt','w');
for i=1:15000
    fprintf(b,'%d \n',estimate(i));
end
fclose(b);
