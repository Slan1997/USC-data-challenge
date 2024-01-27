library(pacman)
pacman::p_load(tidyr, dplyr, readr,tfruns,rappdirs,scales,
               magrittr,zoo,MLmetrics,modeest,tensorflow,keras)

dt = read_csv('summary_all_with_scale.csv') %>% 
  select(id:epoch,contains('sc2')) 
names(dt) = gsub('(eeg_.*)_sc2','\\1',names(dt))

sleep = read_csv('summary_all_no_scale.csv') %>% 
  transmute(sleep = Y)

dt_ready = bind_cols(dt,sleep) %>% mutate(id1 = paste0(id,day)) %>%
  relocate(id1,1)
################### Splits of training/validation/test sets
# The split of train/validation/test sets, denoted as “split,” follows 
# approximately the rule of thumb “80:20”: we randomly select 4 and 2 subjects 
# into train and validation sets (75%) respectively, and 2 subjects (25%) into
# test set. 

seed_split = 890
# student index for train, validation and test, this variable will be used in functions of segment reshaping (see function.R)
idx_files= dt_ready$id1 %>% unique 
set.seed(seed_split)
idx_all = sample(0:7,8)
idx_train = dt_ready %>% filter(id %in% idx_all[1:4]) %>% 
  select(id1) %>% pull %>% unique   ## 4 files for training
idx_val = dt_ready %>% filter(id %in% idx_all[5:6]) %>% 
  select(id1) %>% pull %>% unique  ## 2 for validation
idx_train_val= c(idx_train,idx_val) ## 6 for training and validation
idx_test = dt_ready %>% filter(id %in% idx_all[7:8]) %>% 
  select(id1) %>% pull %>% unique %>% sort
     ## 2 for testing
print(idx_train)
print(idx_val)
print(idx_test)

stepsize=1
m=6
gen_data=gen_split_dt(dt_ready,stepsize=stepsize,m=m) 
save(gen_data,file=paste(m,stepsize,"gen_data.RData",sep = '_'))


load("gen_data.RData")
dt_train=gen_data$dt_train
dt_val=gen_data$dt_val
dt_test=gen_data$dt_test
dt_train_val=gen_data$dt_train_val

train_x = dt_train$dt_x
train_y = dt_train$dt_y %>% to_categorical()

val_x = dt_val$dt_x
val_y = dt_val$dt_y %>% to_categorical()

#checking the dimensions
dim(train_x) 
cat("No of training samples\t",dim(train_x)[[1]],
    "\tNo of validation samples\t",dim(val_x)[[1]])

# > dim(train_x) 
# [1] 23000 5 6 1
# > cat("No of training samples\t",dim(train_x)[[1]],
#       +     "\tNo of validation samples\t",dim(val_x)[[1]])
# No of training samples   23000  No of validation samples         11500> 

# for test subject i in 1:2
test_x_lst = list()
test_y_lst = list()
test_time_lst = list()
len = rep(0,4)  # get total size of each subject's data
for (j in 1:4){
  test_x_lst = c(test_x_lst,list(dt_test[[j]]$dt_x))
  test_y_lst = c(test_y_lst,list(dt_test[[j]]$dt_y %>% to_categorical))
  # get time
  test_time_lst = c(test_time_lst,list(dt_test[[j]]$dt_time))
  
  len[j] = length(test_y_lst[[j]])
}
len
str(test_x_lst)
str(test_y_lst)
str(test_time_lst)



set.seed(seed_split) # 890
tensorflow::tf$random$set_seed(seed_split)
#use_session_with_seed(890,disable_gpu = F,disable_parallel_cpu = F)

model1 <-keras_model_sequential()
#configuring the Model
model1 %>% 
  #### layer1
  layer_conv_2d(filters=16,
                kernel_size=c(3,3),
                padding="same",                
                input_shape=c(5,m,1) ) %>%   
  layer_activation("relu") %>%   # keep unchanged
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(.5) %>% #dropout layer to avoid overfitting
  
  #### layer2
  layer_conv_2d(filter=16,
                kernel_size=c(1,1),
                padding="same") %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(1,1)) %>%
  layer_dropout(.5) %>%
  
  #### output
  layer_flatten() %>%  #flatten the input  
  layer_dense(16) %>% # units: dimensionality of the output space.
  layer_activation("relu") %>%  
  layer_dense(units=3) %>% 
  layer_activation("softmax") 
opt<-optimizer_adam(lr=1e-5)    
model1 %>% compile(loss="categorical_crossentropy",
                   optimizer='adam',
                   metrics = list("AUC","accuracy","Recall","Precision"))
summary(model1)


history1 = model1 %>% fit( train_x,train_y ,batch_size=100,
                           validation_data = list(val_x, val_y),
                           callbacks=callback_early_stopping(monitor = "val_accuracy",
                                                             min_delta = 0,
                                                             patience = 20),
                           epochs=100)

final_result_all = NULL
for (i in 1:length(idx_test)){
  starttime=Sys.time()
  pred = model1 %>% predict(test_x_lst[[i]],test_y_lst[[i]], batch_size = 5)
  endtime=Sys.time()
  comp_cost1 = as.numeric(difftime(endtime,starttime,units="secs"))
  pred_cat = apply(pred,1,which.max) -1
  pred_result = data.frame(pred,pred_cat,time=test_time_lst[[i]])
  dt_ready_sub = dt_ready %>% filter(id1==idx_test[i]) 
  dt_ready_i = dt_ready_sub %>% transmute(true_y=sleep,time=epoch)
  final_result = left_join(dt_ready_i,pred_result,by="time") %>%
    mutate(pred_result_full = na.locf(na.locf(pred_cat,na.rm=F), fromLast = T))
  final_result_all = bind_rows(final_result_all,
                               final_result %>%
                                 mutate(id_test=rep(idx_test[i],nrow(final_result))))
}

final_result_all %<>% mutate_if(is.factor,\(x) as.numeric(x)-1)


### original y
original_y = NULL
file_path = "~/Desktop/USC data challenge/bdhsc_2024/stage1_labeled/"
csv_filenames <- list.files(file_path)[grep('^3|^6', list.files(file_path))]
for (f1 in csv_filenames){
  id = gsub('([0-9])_[0-9].csv','\\1',f1)
  day = gsub('[0-9]_([0-9]).csv','\\1',f1)
  id_test = paste0(id,day)
  dt = read_csv(paste0(file_path,f1))
  y = dt[,ncol(dt)]
  dt1 = cbind(id_test,y) %>% as_tibble %>% rename(y = `5000`)
  original_y = bind_rows(original_y,dt1)
}
save(original_y,file='original_y.RData')
load('original_y.RData')
result_all = bind_cols(original_y,
                       pred_result_full=rep(final_result_all$pred_result_full,each=3),
                       )

## overall accuracy
sum(result_all$y==result_all$pred_result_full)/nrow(result_all)

## each level accuracy
result_all0 = result_all %>% filter(y==0)
sum(result_all0$y==result_all0$pred_result_full)/nrow(result_all0)

result_all1 = result_all %>% filter(y==1)
sum(result_all1$y==result_all1$pred_result_full)/nrow(result_all1)

result_all2 = result_all %>% filter(y==2)
sum(result_all1$y==result_all1$pred_result_full)/nrow(result_all1)




