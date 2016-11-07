function NavieBayes
%% Machine Learning Algorithm:Navie Bayes
%@Author  :  kanadexy
%@Email   :  xiangyustudy@foxmail.com
%@Update  :  2016-11-07 
%if you are something question about the matlab code, please email to me.
%%

%% Origin Data
% 1st-12nd are train data,and rest of them are test data
% age	job     house	credit	class
% 1     0       0       1       0
% 1     0       0       2       0
% 1     1       0       2       1
% 1     1       1       1       1
% 1     0       0       1       0
% 2     0       0       1       0
% 2     0       0       2       0
% 2     1       1       2       1
% 2     0       1       3       1
% 2     0       1       3       1
% 3     0       1       3       1
% 3     0       1       2       1
% =================================
% 3     1       0       2       1
% 3     1       0       3       1
% 3     0       0       1       0
%%

% train_data=[1	0	0	1	;
%             1	0	0	2	;
%             1	1	0	2	;
%             1	1	1	1	;
%             1	0	0	1	;
%             2	0	0	1	;
%             2	0	0	2	;
%             2	1	1	2	;
%             2	0	1	3	;
%             2	0	1	3	;
%             3	0	1	3	;
%             3	0	1	2	];
%         
%         
% train_label=[0 ; 0 ; 1 ; 1 ; 0 ; 0 ; 0 ; 1 ; 1 ; 1 ; 1 ;1];
% 
% test_data=[3	1	0	2	;
%            3	1	0	3	;
%            3	0	0	1	];
       
%% Other Data
% The code can working with the values of attribute are not continuous integer
% you could try!
train_data=[3	-1	10	10	;
            3	-1	10	20	;
            3	7	10	20	;
            3	7	12	10	;
            3	-1	10	10	;
            6	-1	10	10	;
            6	-1	10	20	;
            6	7	12	20	;
            6	-1	12	30	;
            6	-1	12	30	;
            9	-1	12	30	;
            9	-1	12	20	];

train_label=[6 ; 6 ; 9 ; 9 ; 6 ; 6 ; 6 ; 9 ; 9 ; 9 ; 9 ;9];

test_data=[9	7	10	20	;
           9	7	10	30	;
           9	-1	10	10	];
%




      predict_label=NavieBayesClassfiy(train_data,train_label,test_data)
      
end


function predict_label=NavieBayesClassfiy(train_data,train_label,test_data)
    %?Classify?using?NavieBayes?algorithm?
    %?Inputs:?
    %?train_data???-?Train?data ,a N*A matrix.N is the amout of train set,A is the amout of smaple atrributes.?
    %?train_label?   -?Train?data ,a N*1 matrix.N is the amout of train set
    %?test_data??? -?Test?data  ,a M*A matrix.M is the amout of test set,A is the amout of smaple atrributes.?
    %?
    %?Outputs?
    %?predict_label? -?Test?data ,a M*1 matrix.M is the amout of test set?????

    %数据预处理,离散属性数据全部标准为从1开始的增长的整数
    %Preprocessing data,it makes all discrete attributes increse form 1
    [process_train_data,train_data_map]=NormalProcess(train_data); %预处理训练集 Preprocessing train_data
    [process_train_label,label_map]=NormalProcess(train_label);  %预处理训练标签 Preprocessing train_label
     
     [row col]=size(test_data); 
     for i=1:row %预处理测试集 Preprocessing test_data
        for j=1:col   
            %test_data(i,j)
            idx=find(train_data_map(:,j)==test_data(i,j));
            process_test_data(i,j)=idx(1);
        end
     end
     
    

    class_set=tabulate(process_train_label);  %统计出类别个数 Statistic amount of class
    class_prior=class_set(:,3)./100;   %每个类别的先验概率 Get prior probability of every class
    class_frequnce=class_set(:,2);    %每个类别的频数 Get frequnce of every class
    class_num=length(class_frequnce); %类别的个数 Get amout of class
    class_label_unique=class_set(:,1); 
    
    [train_num,atrr_num]=size(process_train_data); %获取训练集属性个数和训练个数 Get amout of train set and amout of smaple atrributes
    
    max_ai_num=0;
    
    for i=1:atrr_num %每个属性有ai个不同的离散值，获取ai中的最大值 Every attribute has ai different discrete value and get the maxiumn one of them.
        
        if(length(unique(train_data(:,i)))>=max_ai_num)
            
            max_ai_num=length(unique(train_data(:,i)));
            
        end
    end
    
    atrr_prio=zeros(atrr_num,max_ai_num*class_num);
    
    
    
    %计算每个属性的先验概率 Get prior probability of every attribute
    for i=1:atrr_num %遍历每一个属性 Traversing every atrribute 
         
        for j=1:class_num %遍历每一个类别 Traversing every class 
            
            train_data_yj=process_train_data(find(process_train_label==class_label_unique(j)),:); %第j个类中测试集所有样本 Get all test samples of j-th class
                     
            
            for k=1:length(unique(train_data(:,i)))
             
                
                atrr_prio(i,(j-1)*max_ai_num+k)=(1+length(find(train_data_yj(:,i)==k)))/(2+class_frequnce(j)); 
                %第j个类别的第i个属性的第k个离散值的个数/第j个类别个数  
                %k-th discrete value of i-th atrribute of j-th class/amoutof j-th class
                %Laplace平滑 Laplace Smoothing
                
                %atrr_prio(i,(j-1)*max_ai_num+k)=length(find(train_data_yj(:,i)==k))/class_frequnce(j);
            end
 
        end
    end
    
     
    [test_num,test_atrr_num]=size(process_test_data);
    
    %预测测试集属于哪个类别 predict 
    for i=1:test_num
        
        max_probability=0; %假设最大后验概率为0  assume the maxiumn posterior probability is 0
        
        for j=1:class_num  
             %计算第j个类的后验概率 Get posterior probability of j-th class
            class_probability=class_prior(j);
            
            for k=1:atrr_num; %依次计算每个属性的在第j类中条件概率 Calculate conditional probability of every attribute which belongs to j-th class
                % atrr_prio(k,(j-1)*max_ai_num+process_test_data(i,k));
                
                class_probability=class_probability*atrr_prio(k,(j-1)*max_ai_num+process_test_data(i,k));
                
            end
            
            
     
    
            if(class_probability>max_probability) %如果当前后验概率最大 if the posterior is maxiumn one
                
                max_probability=class_probability;
                predict_label(i)=j;
                
            end
        end
    end
    
    for i=1:length(label_map)
        
        predict_label(find(predict_label==i))=label_map(i);
        
    end
end

function [process_x,map,map2]=NormalProcess(X)

    [row,col]=size(X);
    process_x=zeros(row,col);
    
    for i=1:col
        
       y=tabulate(X(:,i));
       y(y(:,2)==0,:)=[];
       
       for j=1:length(y(:,1))
           map(j,i)=y(j,1);
       end
     

       for j=1:row
           for k=1:length(y(:,1))
               if(X(j,i)==y(k,1))
                  process_x(j,i)=k; 
                  break; 
               end        
           end
       end
    end
       
 end
    
    
    
