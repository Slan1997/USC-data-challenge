################### Splits of training/validation/test sets
# The split of train/validation/test sets, denoted as “split,” follows 
# approximately the rule of thumb “80:20”: we randomly select 4 and 2 subjects 
# into train and validation sets (75%) respectively, and 2 subjects (25%) into
# test set. 

seed_split = 890
# student index for train, validation and test, this variable will be used in functions of segment reshaping (see function.R)
idx_files=0:7
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


hyper = read_csv(paste0(hyperpara_path,"testhyper_full_ver_4L.csv")) %>% filter(epoch_type==epoch1 & hr== (h==1) )%>%
  select(-l)
al = read_csv(paste0(path_fs,"alignment_full_ver.csv")) %>% rename(epoch_type=epoch) #read_csv(paste0(path_fs,"alignment_full_ver.csv")) %>% rename(epoch_type=epoch)
l_var_lst = al %>% filter(seed_split==seed_split_val&epoch_type==epoch1 & hr== (h==1)&align==hyper$align) %>% select(seed_split,l,var_lst)

hyper = hyper%>% bind_cols(l_var_lst)#  HR
hyper %>% as.data.frame
dim(hyper)


##############
#tag=read_csv(paste0(tag_path,"/sleep_tag.csv"))


###################################################  cnn part
i = 1 # best config_idx, no need to specify, since hyper only has one row

hyper[i,]
# Building the LSTM model
# flags {tfruns}
FLAGS = get_flags(hyper) 

#### read in data
epoch_ty = which(epoch_type==hyper$epoch_type[i])
time = read_csv(paste0(proc_path[epoch_ty],sec[epoch_ty],"_ready.csv")) %>%  ##col_types = cols(calender_time=col_datetime())) %>%
  select(unix_sec)

var_lst = c("e4_id", unlist(str_split(FLAGS$var_lst,',')),"sleep")
modal = unlist(str_split(FLAGS$modality,','))

dt_ready = read_csv(paste0(proc_path[epoch_ty],sec[epoch_ty],"_ready_norm.csv")) %>%
  dplyr::select(all_of(var_lst)) %>% select(e4_id,starts_with(modal),sleep) %>% 
  mutate_at(c("hourofday","age"),funs(scale)) %>%
  bind_cols(time)  

#### get train/test split
###create train, validation and test set
# student index for train, validation and test
idx_files=1:25
set.seed(FLAGS$seed)
idx_all = sample(c(1,3:25),24)
idx_train = idx_all[1:15]   ## 15 files for training
idx_val = idx_all[16:20]    ## 5 for validation
idx_train_val=idx_all[1:20] ## 20 for training and validation
idx_test = sort(idx_all[21:24])   ## 4 for testing
print(idx_train)
print(idx_val)

#gen_split_dt = function(dt_ready,stepsize,var_lst,m,hr=T,LSTM=F)
gen_data=gen_split_dt(dt_ready,FLAGS$step_size,var_lst,FLAGS$m,LSTM=T) # handled 22, use hr=T here, actually var_lst already consider hr scenario

dt_train=gen_data$dt_train
dt_val=gen_data$dt_val
dt_test=gen_data$dt_test
dt_train_val=gen_data$dt_train_val

train_x = dt_train$dt_x
train_y = dt_train$dt_y

val_x = dt_val$dt_x
val_y = dt_val$dt_y

#checking the dimensions
dim(train_x) 
cat("No of training samples\t",dim(train_x)[[1]],
    "\tNo of validation samples\t",dim(val_x)[[1]])

# > dim(train_x) 
# [1] 4091   18   10
# > cat("No of training samples\t",dim(train_x)[[1]],
#       +     "\tNo of validation samples\t",dim(val_x)[[1]])
# No of training samples	 4091 	No of validation samples	 1305> 

# for test subject i in 1:5
test_x_lst = list()
test_y_lst = list()
test_time_lst = list()
len = rep(0,4)  # get total size of each subject's data
for (j in 1:4){
  test_x_lst = c(test_x_lst,list(dt_test[[j]]$dt_x))
  test_y_lst = c(test_y_lst,list(dt_test[[j]]$dt_y))
  # get time
  test_time_lst = c(test_time_lst,list(dt_test[[j]]$dt_time))
  
  len[j] = nrow(test_y_lst[[j]])
}
len
str(test_x_lst)
str(test_y_lst)
str(test_time_lst)


middle_value = function(x) apply(x,1,function(z) z[ceiling(FLAGS$m/2)])
test_time_lst1 = lapply(test_time_lst,middle_value)
str(test_time_lst1)
test_time=unlist(test_time_lst1)
str(test_time)


###### will be used in getting win_size, not sorted!!!!
train_val_x_lst = list()
train_val_y_lst = list()
len_train_val=rep(0,20)
for (j in 1:20){
  train_val_x_lst = c(train_val_x_lst,list(dt_train_val[[j]]$dt_x))
  train_val_y_lst = c(train_val_y_lst,list(dt_train_val[[j]]$dt_y))
  len_train_val[j] = nrow(train_val_y_lst[[j]])
}
len_train_val
cum_len=c(0,cumsum(len_train_val))
cum_len


#######
skip=T
if (!skip){
  
  
  # train LSTM
  # specify hyperparameter
  if (epoch_ty==3){
    num_units = 16
  }else num_units = 32
  
  library(tensorflow)
  library(keras)
  lstm_model <- keras_model_sequential()
  lstm_model %>%
    bidirectional(layer_lstm(batch_input_shape = dim(train_x),units = num_units,
                             dropout = 0.2,  return_sequences = TRUE
                             #units: size of the layer
                             #input_shape = c(251,18,10), # batch size, timesteps, features
    )) %>%
    layer_dense(units = 1, activation = "sigmoid")
  opt<-optimizer_adam(lr=1e-5)    # lr must be this small
  lstm_model %>% compile(loss="binary_crossentropy",
                         optimizer=opt,
                         metrics = list("AUC","accuracy","Recall","Precision"))
  
  starttime=Sys.time()
  history_lstm = lstm_model  %>% fit(
    x = train_x,y = train_y,
    validation_data = list(val_x, val_y),
    batch_size = FLAGS$batch_size,
    callbacks=callback_early_stopping(monitor = "val_accuracy",
                                      min_delta = 0,
                                      patience = FLAGS$patience),
    epochs = FLAGS$num_epoch
  )
  endtime=Sys.time()
  comp_train_lstm = as.numeric(difftime(endtime,starttime,units="secs"))
  summary(lstm_model)
  lstm_model %>% save_model_hdf5(paste0("~/E4_CNN_2022/saved_test_models/lstm_",index,".h5"))
  
  #lstm_model <- load_model_hdf5(paste0("~/E4_CNN_2022/saved_test_models/lstm_",index,".h5"))
  summary(lstm_model)
  
  all_x=abind::abind(train_x,val_x,along = 1)
  all_y=rbind(train_y,val_y)
  
  all_y = as.numeric(rowSums(all_y) > (FLAGS$m/2) )
  str(all_y)
  
  pred_train_values = lstm_model %>% predict(all_x,all_y, batch_size = 5)
  # dim(pred_train_values)
  # [1] 5396   18    1
  pred_train_values = pred_train_values[,,1]
  str(pred_train_values)
  bin_pred_train_values = as.numeric(rowSums(pred_train_values>.5)>(FLAGS$m/2) )
  str(bin_pred_train_values)
  
  # train pred values
  data.frame(pred_binary=bin_pred_train_values,true_y=all_y) %>% 
    write_csv(paste0("~/E4_CNN_2022/pred_values/lstm_trainval_",index,'.csv'))
}


