clear all;
clc;
load TrainDataWithFragmentSequence

fragmentSequence=fragmentSequenceTrain;
ptmNonPtmSite=ptmNonPtmSiteTrain;

indexPositiveSite=ismember(ptmNonPtmSite, '1');
indexNegativeSite=~indexPositiveSite;

fragmentSequencePositive=fragmentSequence(indexPositiveSite,1);
fragmentSequenceNegative=fragmentSequence(indexNegativeSite,1);


fragmentSequencePositiveString=char(fragmentSequencePositive);
fragmentSequenceNegativeString=char(fragmentSequenceNegative);


windowSize=20;
middle=windowSize+1;
fragmentLength=(2*windowSize)+1;

dataSize=size(ptmNonPtmSite,1);
probabilityFeature=zeros(dataSize,fragmentLength-1); 

for sN=1:dataSize
    
    oneSample=fragmentSequence{sN,1};
    oneSampleProbabilityFeature=zeros(fragmentLength-1,1);
    
    featureNo=1;
    
    for i=1:fragmentLength
        
        if(i>=middle-1 && i<=middle+1) 
            
            if (i==middle-1 || i==middle+1)
                charA=oneSample(i);
                %find conditional probability from positive set
                columnPositiveA=fragmentSequencePositiveString(:,i);
                numCharA=sum(ismember(columnPositiveA, charA));
                positiveSize=size(fragmentSequencePositive,1);
                conProbabilityPositive=numCharA/positiveSize;
                
                %find conditional probability from negative set
                columnNegativeA=fragmentSequenceNegativeString(:,i);
                numCharA=sum(ismember(columnNegativeA, charA));
                negativeSize=size(fragmentSequenceNegative,1);
                conProbabilityNegative=numCharA/negativeSize;
                
                oneSampleProbabilityFeature(featureNo,1)=conProbabilityPositive-conProbabilityNegative;
                featureNo=featureNo+1;
            end

        else
            
            if (i<middle)
                charA=oneSample(i);
                charB=oneSample(i+1);
                
                colNoA=i;
                ColNoB=i+1;
                
            else
                charA=oneSample(i);
                charB=oneSample(i-1);
                
                colNoA=i;
                ColNoB=i-1;
            end
            
            %find conditional probability from positive set
            
            columnPositiveA=fragmentSequencePositiveString(:,colNoA);
            columnPositveB=fragmentSequencePositiveString(:,ColNoB);
   
            indexCharBInColPositiveB=ismember(columnPositveB,charB);
            
            charBPresentInColPositiveA=columnPositiveA(indexCharBInColPositiveB);
            
            indexCharAInCharBPresentInColPositiveA=ismember(charBPresentInColPositiveA,charA);
            
            numCharB=sum(indexCharBInColPositiveB);
            numCharA=sum(indexCharAInCharBPresentInColPositiveA);
            
            if (numCharB==0)
                conProbabilityPositive=0;
            else
                conProbabilityPositive=numCharA/numCharB;
            end
            
            %find conditional probability from negative set
            
            columnNegativeA=fragmentSequenceNegativeString(:,colNoA);
            columnNegatveB=fragmentSequenceNegativeString(:,ColNoB);
   
            indexCharBInColNegativeB=ismember(columnNegatveB,charB);
            
            charBPresentInColNegativeA=columnNegativeA(indexCharBInColNegativeB);
            
            indexCharAInCharBPresentInColNegativeA=ismember(charBPresentInColNegativeA,charA);
            
            numCharB=sum(indexCharBInColNegativeB);
            numCharA=sum(indexCharAInCharBPresentInColNegativeA);
            
            if (numCharB==0)
                conProbabilityNegative=0;
            else
                conProbabilityNegative=numCharA/numCharB;
            end
                                              
            oneSampleProbabilityFeature(featureNo,1)=conProbabilityPositive-conProbabilityNegative;            
            featureNo=featureNo+1;
            
        end
    
    end
    
    probabilityFeature(sN,:)=oneSampleProbabilityFeature;
    disp(sN)
end


%probabilityFeature_Car_K=probabilityFeature;

save Prob_Feature.mat probabilityFeature
csvwrite('Prob_Feature.csv',probabilityFeature);