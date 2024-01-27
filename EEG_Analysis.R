library(pacman)
pacman::p_load(tidyr, dplyr, readr,tfruns,rappdirs,scales,
               magrittr,zoo,MLmetrics,modeest,tensorflow,keras,
               multidplyr,tidyverse,data.table)

######################################################
################### Data preprocessing
csv_filenames <- list.files("bdhsc_2024/stage1_labeled/")
for(i in seq_along(csv_filenames)){
  assign(paste0("dt",str_sub(basename(csv_filenames),1,3)[i]), 
         fread(paste0("bdhsc_2024/stage1_labeled/",csv_filenames[i]), skip = 1))
}
for (i in str_sub(basename(csv_filenames),1,3)){
  assign(paste0("y",i), get(paste0("dt",i))[,5001])
}
for (i in str_sub(basename(csv_filenames),1,3)){
  assign(paste0("x",i), get(paste0("dt",i))[,1:5000])
}
calc_mode <- function(codes){
  which.max(tabulate(codes))
}
sum_all <- NULL
for (i in str_sub(basename(csv_filenames),1,3)[-1]){
  df_temp <- get(paste0("x",i))
  y_temp <- get(paste0("y",i))
  df_temp_1 <- df_temp[seq(1,8640, 3),]
  y_temp_1 <- y_temp[seq(1,8640, 3),]
  df_temp_2 <- df_temp[seq(2,8640, 3),]
  y_temp_2 <- y_temp[seq(2,8640, 3),]
  df_temp_3 <- df_temp[seq(3,8640, 3),]
  y_temp_3 <- y_temp[seq(3,8640, 3),]
  df_wide_temp <- cbind(df_temp_1, df_temp_2, df_temp_3)
  y_wide_temp <- cbind(y_temp_1, y_temp_2, y_temp_3)
  df_wide_temp[,"epoch"] <- seq(1,2880,1)
  y_wide_temp[,"epoch"] <- seq(1,2880,1)
  names(df_wide_temp) <- c(paste0("V",1:15000),"epoch")
  names(y_wide_temp) <- c(paste0("y",1:3),"epoch")
  df_sum_temp <- df_wide_temp %>%
    pivot_longer(!epoch,names_to = "var",values_to = "value")%>%
    group_by(epoch) %>%
    summarise(eeg_mean = mean(value),
              eeg_sd = sd(value),
              eeg_q1 = quantile(value,probs = 0.25),
              eeg_q2 = quantile(value,probs = 0.5),
              eeg_q3 = quantile(value,probs = 0.75))%>%
    ungroup()%>%
    mutate(id = str_sub(i,1,1),
           day = str_sub(i,3,3))
  y_sum_temp <- y_wide_temp %>%
    pivot_longer(!epoch,names_to = "y_name",values_to = "value")%>%
    group_by(epoch) %>%
    summarise(Y = mfv(value))%>%
    ungroup()%>%
    mutate(id = str_sub(i,1,1),
           day = str_sub(i,3,3))
  df_sum_temp_metric <- df_sum_temp[,c("id","day","epoch","eeg_mean","eeg_sd","eeg_q1","eeg_q2","eeg_q3")] %>%
    full_join(y_sum_temp, by = c("id", "day","epoch"))
  assign(paste0("sum",i), df_sum_temp_metric)
  sum_all <- rbind(sum_all, df_sum_temp_metric)
}
fwrite(sum_all,"summary_all_no_scale.csv")

# sum_all <- fread("summary_all_no_scale.csv")

norm1 = function(x) return((x-mean(x))/sd(x))

scale_within_person <- sum_all %>%
  group_by(id) %>%
  mutate(eeg_mean_sc1 = norm1(eeg_mean),
         eeg_sd_sc1 = norm1(eeg_sd),
         eeg_q1_sc1 = norm1(eeg_q1),
         eeg_q2_sc1 = norm1(eeg_q2),
         eeg_q3_sc1 = norm1(eeg_q3)) %>%
  ungroup()%>%
  mutate(eeg_mean_sc2 = norm1(eeg_mean_sc1),
         eeg_sd_sc2 = norm1(eeg_sd_sc1),
         eeg_q1_sc2 = norm1(eeg_q1_sc1),
         eeg_q2_sc2 = norm1(eeg_q2_sc1),
         eeg_q3_sc2 = norm1(eeg_q3_sc1))
fwrite(scale_within_person,"summary_all_with_scale.csv")

######################################################
################### Model training/testing
####### FUNCTIONS NEEDED
### reshaping time series into image
# ts: original time series
# len_seg: m, the total number of epochs in a synthesized image
# stepsize: step size of the sliding window

build_matrix <- function(ts, len_seg,stepsize){ 
  sapply(seq(1,(length(ts) - len_seg + 1),stepsize), 
         function(x) ts[x:(x + len_seg - 1)])
}
#build_matrix(1:10,4,2)

reformat = function(dt_ready,idx_files,m,stepsize){
  fi_reform_all = NULL 
  # each subject's images are generated independently, 
  # then all subjects' images are combined into one matrix
  for (d in idx_files){
    fi =filter(dt_ready,id1==d)
    #n = 2880
    idx = build_matrix(1:2880,m,stepsize) %>% as.vector()
    dt = fi %>% slice(idx) 
    dt = dt %>% mutate(pt_idx = rep(1:m,nrow(dt)/m))
    fi_reform_all = bind_rows(fi_reform_all,dt)
  }
  return(fi_reform_all)
}


### split train and test
# input format must be: 
# the first column should be e4_id, and the last two column are: sleep, pt_idx.
gen_final_dt = function(fi_reform_all,idx_files,idx_trte,m,l){
  fi_reform = filter(fi_reform_all,id1 %in% idx_trte)
  n = nrow(fi_reform)/m
  dt_x = NULL  
  
  for (j in 1:m){
    dt_nl = fi_reform %>% filter(pt_idx==j) %>% 
      dplyr::select(contains("eeg"))%>% as.matrix() %>% as.vector()
    dt_x = c(dt_x,dt_nl)
  }
  #(batch_size, height, width, channels)
  dim(dt_x) = c(n,l,m,1)
  
  dt_y = fi_reform %>% dplyr::select(sleep) %>%
    mutate(voter = rep(1:n,each=m)) %>%
    group_by(voter) %>% summarise(s=mfv(sleep)[1])%>% pull(s)
  #dt_y
  
  dt_time_origin = fi_reform %>% dplyr::select(epoch) %>%  
    mutate(time_chunk = rep(1:n,each=m))
  dt_time = dt_time_origin %>%
    group_by(time_chunk) %>% 
    summarise(t = epoch[ceiling(m/2)]) %>% pull(t)
  # use the m/2 th time point of each time chunk as the true time for that window
  return(list(dt_x=dt_x,dt_y=dt_y,dt_time=dt_time))
}



## reshape part for LSTM
reshape_LSTM = function(dt_ready,idx_trte,m,l,stepsize){
  for (idx in 1:length(idx_trte)){#length(idx_files)
    id0 = idx_trte[idx] #
    fi =filter(dt_ready,id1==id0)
    
    dt_tran = lapply(fi,function(x){
      t(sapply(seq(1,(length(x) - m + 1),stepsize), # (1:20)-5-5+1
               function(z) x[z:(z + m - 1)])) } )
    
    if (idx==1){
      dt1_tran = dt_tran
    }else{
      for (li in 1:length(dt1_tran)) dt1_tran[[li]] = rbind(dt1_tran[[li]],dt_tran[[li]])
    }
  }
  
  x_names = names(dt1_tran)[- grep('^eeg',names(dt1_tran))]
  tran_data_x = do.call(bind_cols,dt1_tran[x_names])
  # now we transform it into 3Dim form
  tran_arr_x <- array(
    data = as.numeric(unlist(tran_data_x)),
    dim = c( nrow(tran_data_x), m,l))
  #dim(tran_arr_x)
  
  tran_data_y = dt1_tran[['sleep']]
  tran_data_time = dt1_tran[['epoch']] # later using ceiling m/2 as the time for this point????
  #dim(tran_data_y); dim(tran_data_time)
  return(list(dt_x = tran_arr_x,dt_y=tran_data_y,dt_time=tran_data_time))
}


### gen_split_dt(): final function for generating reformed train/test/validation dataset
gen_split_dt = function(dt_ready,stepsize,m,LSTM=F){ 
  l = 5
  if (LSTM==F){
    fi_reform_all = reformat(dt_ready,idx_files,m,stepsize)
    dt_train = gen_final_dt(fi_reform_all,idx_files,idx_train,m,l)
    dt_val = gen_final_dt(fi_reform_all,idx_files,idx_val,m,l)
    
    # test should by individual
    dt_test = list()
    for (k in 1:length(idx_test)){
      dt_test = c(dt_test,list(gen_final_dt(fi_reform_all,idx_files,idx_test[k],m,l)))
    }
    names(dt_test) = paste0("subject_",idx_test)
    
    dt_train_val =  list()
    
    for (k in 1:length(idx_train_val)){
      dt_train_val = c(dt_train_val,list(gen_final_dt(fi_reform_all,idx_files,idx_train_val[k],m,l)))
    }
    names(dt_train_val) = paste0("subject_",idx_train_val)
    
  }else{ ## for LSTM
    dt_train = reshape_LSTM(dt_ready,idx_train,m,l,stepsize)
    dt_val = reshape_LSTM(dt_ready,idx_val,m,l,stepsize)
    
    # test should by individual
    dt_test = list()
    for (k in 1:length(idx_test)){
      dt_test = c(dt_test,list( reshape_LSTM(dt_ready,idx_test[k],m,l,stepsize) ))
    }
    names(dt_test) = paste0("subject_",idx_test)
    
    dt_train_val = list()
    for (k in 1:length(idx_train_val)){
      dt_train_val = c(dt_train_val,list(reshape_LSTM(dt_ready,idx_train_val[k],m,l,stepsize) ))
    }
    names(dt_train_val) = paste0("subject_",idx_train_val)
  }
  
  return(list(dt_train=dt_train,dt_val=dt_val,dt_test=dt_test,dt_train_val=dt_train_val))
}


################### Read in the preprocessed data
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
# into train and validation sets (75% in total) respectively, and 2 subjects (25%) into
# test set. 

seed_split = 890
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

stepsize=1 # 2
m=6 # 12, 20
gen_data=gen_split_dt(dt_ready,stepsize=stepsize,m=m) 
# save(gen_data,file="gen_data.RData")
# load("gen_data.RData")

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

# for each test subject i, we can fit the model separately, 
# in case we want to evaluate model's classification on each person.
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
  layer_dense(units=3) %>%  # 3 classes
  layer_activation("softmax") 
opt<-optimizer_adam(lr=1e-5)    # 1e-4
model1 %>% compile(loss="categorical_crossentropy",
                   optimizer='adam',
                   metrics = list("AUC","accuracy","Recall","Precision"))
summary(model1)

starttime=Sys.time()
history1 = model1 %>% fit( train_x,train_y ,batch_size=100,
                           validation_data = list(val_x, val_y),
                           callbacks=callback_early_stopping(monitor = "val_accuracy",
                                                             min_delta = 0,
                                                             patience = 20),
                        epochs=100)
endtime=Sys.time()
as.numeric(difftime(endtime,starttime,units="secs"))
final_result_all = NULL
for (i in 1:length(idx_test)){
  #starttime=Sys.time()
  pred = model1 %>% predict(test_x_lst[[i]],test_y_lst[[i]], batch_size = 5)
  #endtime=Sys.time()
  #comp_cost1 = as.numeric(difftime(endtime,starttime,units="secs"))
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
#save(original_y,file='original_y.RData')
#load('original_y.RData')
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
sum(result_all2$y==result_all2$pred_result_full)/nrow(result_all2)



test_result = result_all %>% 
  transmute(id_test,animal_id=gsub('([0-9])[0-9]','\\1',id_test),
            record_id=gsub('[0-9]([0-9])','\\1',id_test),
            pred_class = pred_result_full)

test_result %>% write_csv('Team Bear_Round 1 Analysis Results_TestResult.csv')
