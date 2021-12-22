fidForRead = fopen('pos.txt');

allRecordPositive=textscan(fidForRead,'%s');
fragmentSequenceTrainPositive=allRecordPositive{1, 1};
positiveSize=size(fragmentSequenceTrainPositive,1);
pupNonPupSiteTrainPositive=cell(positiveSize,1);
defaultValue={'1'};
pupNonPupSiteTrainPositive(:,1)=defaultValue;
fclose(fidForRead);


fidForRead = fopen('neg.txt');
allRecordNegative=textscan(fidForRead,'%s');
fragmentSequenceTrainNegative=allRecordNegative{1, 1};
negativeSize=size(fragmentSequenceTrainNegative,1);
pupNonPupSiteTrainNegative=cell(negativeSize,1);
defaultValue={'-1'};
pupNonPupSiteTrainNegative(:,1)=defaultValue;
fclose(fidForRead);

fragmentSequenceTrain=[fragmentSequenceTrainPositive; fragmentSequenceTrainNegative];
ptmNonPtmSiteTrain=[pupNonPupSiteTrainPositive; pupNonPupSiteTrainNegative];

save TrainDataWithFragmentSequence.mat  fragmentSequenceTrain ptmNonPtmSiteTrain


