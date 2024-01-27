################### Splits of training/validation/test sets
# The split of train/validation/test sets, denoted as “split,” follows 
# approximately the rule of thumb “80:20”: we randomly select 4 and 2 subjects 
# into train and validation sets (75%) respectively, and 2 subjects (25%) into
# test set. 

seed_split = 890
# student index for train, validation and test, this variable will be used in functions of image reshaping (see function.R)
idx_files=0:7
# originally, there are 25 students' data, however student 2's data is
# not qualified for the experiment. So only 24 students' data were used in the experiment.
set.seed(seed_split)
idx_all = sample(0:7,8)
idx_train = idx_all[1:4]   ## 4 files for training
idx_val = idx_all[5:6]    ## 2 for validation
idx_train_val=idx_all[1:6] ## 6 for training and validation
idx_test = sort(idx_all[7:8])   ## 2 for testing
print(idx_train)
print(idx_val)
print(idx_test)


# split data for train_val and test
dt_train =  dt_dum %>% filter(id %in% idx_train_val) # check the order of e4_id
dt_test = dt_dum %>% filter(id %in% idx_test)


# correlation and importance rank to decide alignment
features = dt_train%>% select(eeg_mean:eeg_q3) 

correlationMatrix <- cor(features)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75,names = T)

# print indexes of highly correlated attributes
print(highlyCorrelated)
f = DF2formula(dt_dum[,!(colnames(dt_dum) %in% c('e4_id',highlyCorrelated))]) # remove e4_id
#f = DF2formula(dt_dum[,-2])
f

set.seed(12345)
RFModel <- randomForest(f, data = dt_train, importance=T)
importance(RFModel)
#varImpPlot(RFModel)
im = as.data.frame(importance(RFModel)) %>% arrange(desc(MeanDecreaseAccuracy))
im$feature = rownames(im)
im$rank = 1:nrow(im)




