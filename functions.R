# Functions needed for the experiment.

###################### Workflow Implementation

# feature selection and alignment
get_final_rank = function(combined_rank,seedsplit = c(890, 9818,7,155,642),
                          align=list(c(2,1,4,3),c(4,1,2,3),c(1,3,2,4),c(4,3,1,2),c(3,1,2,4))){
  fes = unique(combined_rank$feature)
  full = NULL
  for (i in 1:length(fes)){
    sub = combined_rank %>% filter(feature==fes[i]) %>% 
      group_by(seed_split) %>% summarise(s = sum(rank)) %>% ungroup %>% 
      mutate(feature=fes[i]) 
    full= bind_rows(full,sub)
  }
  
  imp_rank_feature = tibble()
  for (seed in seedsplit){
    tb = full %>% filter(seed_split==seed) %>% arrange(s) 
    imp_rank_feature = bind_rows(imp_rank_feature,tb)
  }
  #imp_rank_feature
  
  al = expand.grid(seed_split=seedsplit,align=1:length(align))
  alm = NULL # alignment
  fea_lst = NULL
  l = NULL # number of features
  for (i in 1:nrow(al)){
    sub = imp_rank_feature %>% filter(seed_split ==al$seed_split[i]) 
    fe = sub %>% dplyr::select(feature) %>% pull
    l = c(l, length(fe))
    fea_lst = c(fea_lst,paste(fe,collapse=','))
    modality = fe[grep('[_]',fe)] # features of modalities
    demo = fe[-grep('[_]',fe)]  # demographic features
    modal = sort(unique(gsub('(^[a-z]+)_[a-z]+.*','\\1',modality))) #"acc"  "eda"  "hr"   "temp"
    sub_alm = modal[align[[al$align[i]]]] # arrange the modality as in the corresponding alignment.
    alm = c(alm,paste(c(demo, sub_alm[!is.na(sub_alm)]),collapse=','))
  }
  al$l = l
  al$modal = alm
  al$var_lst = fea_lst
  return(al)
}




### reshaping time series into image
# ts: original time series
# len_seg: m, the total number of epochs in a synthesized image
# stepsize: step size of the sliding window

build_matrix <- function(ts, len_seg,stepsize){ 
  sapply(seq(1,(length(ts) - len_seg + 1),stepsize), 
         function(x) ts[x:(x + len_seg - 1)])
}
build_matrix(1:10,4,2)

reformat = function(dt_ready,idx_files,m,stepsize){
  fi_reform_all = NULL 
  # each subject's images are generated independently, 
  # then all subjects' images are combined into one matrix
  for (id in idx_files){
    fi =filter(dt_ready,e4_id==id)
    
    if (id !=22){
      n = nrow(fi)
      idx = build_matrix(1:n,m,stepsize) %>% as.vector()
      dt = fi %>% slice(idx) 
      dt = dt %>% mutate(pt_idx = rep(1:m,nrow(dt)/m))
      
    }else{ # handle subject 22
      fi1 = fi %>% filter(unix_sec>=1582124305 & unix_sec<=1582262785) 
      n1 = nrow(fi1)
      idx1 = build_matrix(1:n1,m,stepsize) %>% as.vector()
      dt1 = fi1 %>% slice(idx1) 
      dt1 = dt1 %>% mutate(pt_idx = rep(1:m,nrow(dt1)/m))
      
      fi2 = fi %>% filter(unix_sec>=1582316887 & unix_sec<=1582432223) 
      n2 = nrow(fi2)
      idx2 = build_matrix(1:n2,m,stepsize) %>% as.vector()
      dt2 = fi2 %>% slice(idx2) 
      dt2 = dt2 %>% mutate(pt_idx = rep(1:m,nrow(dt2)/m))
      
      dt = bind_rows(dt1,dt2)
    }
    fi_reform_all = bind_rows(fi_reform_all,dt)
  }
  return(fi_reform_all)
}


### split train and test
# input format must be: 
# the first column should be e4_id, and the last two column are: sleep, pt_idx.
gen_final_dt = function(fi_reform_all,idx_files,idx_trte,m,l){
  fi_reform = filter(fi_reform_all,e4_id %in% idx_files[idx_trte])
  n = nrow(fi_reform)/m
  dt_x = NULL  
  
  for (j in 1:m){
    dt_nl = fi_reform %>% filter(pt_idx==j) %>% 
      dplyr::select(2:(ncol(fi_reform_all)-3) )%>% as.matrix() %>% as.vector()
    dt_x = c(dt_x,dt_nl)
  }
  #(batch_size, height, width, channels)
  dim(dt_x) = c(n,l,m,1)
  
  dt_y = fi_reform %>% dplyr::select(sleep) %>%
    mutate(voter = rep(1:n,each=m)) %>%
    group_by(voter) %>% summarise(s=sum(sleep)) %>% pull(s)
  dt_y[dt_y<(m/2)] = 0 # 0 for wake
  dt_y[dt_y>=(m/2)] = 1  # 1 for sleep
  #dt_y
  
  dt_time_origin = fi_reform %>% dplyr::select(unix_sec) %>%  
    mutate(time_chunk = rep(1:n,each=m))
  dt_time = dt_time_origin %>%
    group_by(time_chunk) %>% 
    summarise(t = unix_sec[ceiling(m/2)]) %>% pull(t)
  # use the m/2 th time point of each time chunk as the true time for that window
  return(list(dt_x=dt_x,dt_y=dt_y,dt_time=dt_time))
}



## reshape part for LSTM
reshape_LSTM = function(dt_ready,idx_trte,m,l,stepsize){
  for (idx in 1:length(idx_trte)){#length(idx_files)
    id = idx_trte[idx] # 24
    fi =filter(dt_ready,e4_id==id) 
    if (id !=22){
      # fi = fi %>% dplyr::select(-e4_id,-unix_sec)
      dt_tran = lapply(fi,function(x){
        t(sapply(seq(1,(length(x) - m + 1),stepsize), # (1:20)-5-5+1
                 function(z) x[z:(z + m - 1)])) } )
    }else{
      fi1 = fi %>% filter(unix_sec>=1582124305 & unix_sec<=1582262785) # %>%
      #dplyr::select(-e4_id,-unix_sec)
      dt_tran = lapply(fi1,function(x){
        t(sapply(seq(1,(length(x) - m + 1),stepsize), # (1:20)-5-5+1
                 function(z) x[z:(z + m - 1)])) } )
      
      fi2 = fi %>% filter(unix_sec>=1582316887 & unix_sec<=1582432223) #%>%
      #dplyr::select(-e4_id,-unix_sec)
      dt_tran_part2 = lapply(fi2,function(x){
        t(sapply(seq(1,(length(x) - m + 1),stepsize), # (1:20)-5-5+1
                 function(z) x[z:(z + m - 1)])) } )
      
      for (li in 1:length(dt_tran)) dt_tran[[li]] = rbind(dt_tran[[li]],dt_tran_part2[[li]])
    }
    
    if (idx==1){
      dt1_tran = dt_tran
    }else{
      for (li in 1:length(dt1_tran)) dt1_tran[[li]] = rbind(dt1_tran[[li]],dt_tran[[li]]) 
    }
  }
  
  x_names = names(dt1_tran)[-which(names(dt1_tran) %in% c("e4_id", "sleep", "unix_sec") )]
  tran_data_x = do.call(bind_cols,dt1_tran[x_names])
  # now we transform it into 3Dim form
  tran_arr_x <- array(
    data = as.numeric(unlist(tran_data_x)),
    dim = c( nrow(tran_data_x), m,l))
  #dim(tran_arr_x)
  
  tran_data_y = dt1_tran[['sleep']]
  tran_data_time = dt1_tran[['unix_sec']] # later using ceiling m/2 as the time for this point????
  #dim(tran_data_y); dim(tran_data_time)
  return(list(dt_x = tran_arr_x,dt_y=tran_data_y,dt_time=tran_data_time))
}


### gen_split_dt(): final function for generating reformed train/test/validation dataset
gen_split_dt = function(dt_ready,stepsize,var_lst,m,hr=T,LSTM=F){ # hr = 1 or 0 (have or not have) # var_lst not using here.
  l = ncol(dt_ready)-3
  
  if (LSTM==F){
    if (hr==F) dt_ready = dt_ready %>% dplyr::select(-(hr_mean:hr_q3))
    
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

########### Training => find the best Configuration j*

#### summary_over_seed(): find the max or mean across seed value for each metric
# summary_type="mean" or "max"
summary_over_seed = function(dt,num_seed,summary_type="mean"){   # default is mean
  comb_seed = dt %>% mutate(config_idx = rep(1: (nrow(dt)/num_seed),each=num_seed))
  if (summary_type=="max"){
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), max)
  }else{
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), mean)
  }
  return(over_seed)
}

#### Get training result table
# dt: metric_tb; 
# l: number of layer; 
# EDA_type="cat 3" or "median"; 
# summary_type="mean" or "max"

# not using anymore
train_result_tb = function(dt,hyper,l,num_seed,EDA_type="cat 3",summary_type="mean"){
  if (l==1){
    hyper_seedset = hyper %>% dplyr::select(-2) %>% slice(seq(1,nrow(hyper),num_seed)) %>% 
      mutate(config_idx = 1:(nrow(hyper)/num_seed)) %>% dplyr::select(18,1:17)
  }else{
    hyper_seedset = hyper %>% dplyr::select(-2) %>% slice(seq(1,nrow(hyper),num_seed)) %>% 
      mutate(config_idx = 1:(nrow(hyper)/num_seed)) %>% dplyr::select(25,1:24)
  }
  comb_seed = dt %>% mutate(config_idx = rep(1: (nrow(dt)/num_seed),each=num_seed))
  if (summary_type=="max"){
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), max)
  }else{
    over_seed = comb_seed %>% group_by(config_idx) %>% 
      summarise_at(vars(val_accuracy:val_AUC), mean)
  }
  criterias=c("val_accuracy","val_recall","val_precision","val_AUC")
  config_tb = NULL
  criteria_tb = NULL
  for (j in 2:5){
    config_num = which.max(over_seed[,j]%>%pull)
    best_config = hyper_seedset[config_num,]
    config_tb = bind_rows(config_tb,best_config)
    criteria_tb = bind_rows(criteria_tb,
                            over_seed[config_num,] %>% mutate(EDA_type,criteria=criterias[(j-1)]) )
  }
  return(list(best_configs=config_tb,metrics=criteria_tb))
}

#### Test
get_flags = function(hyper,i=1){
  if (hyper$conv_layer=="1 layer"){
    FLAGS <- flags(
      flag_string("modality",hyper$modal[i]),
      flag_string("var_lst",hyper$var_lst[i]),
      
      flag_string("epoch",hyper$epoch_type[i]),
      # seed for train/test split and cnn
      flag_integer("seed",hyper$seed_split[i]),
      
      # m
      flag_integer("m", hyper$m[i]),
      # l
      flag_integer("l", hyper$l[i]),
      # stepsize
      flag_integer("step_size", hyper$step_size[i]),
      # hr indicator
      flag_boolean("hr", hyper$hr[i]),
      
      # kernel size: h*w
      flag_integer("kernel_size_h", hyper$kernel_size_h1[i]),
      flag_integer("kernel_size_w", hyper$kernel_size_w1[i]),
      # number of kernels
      flag_integer("num_kernel", hyper$num_kernel1[i]),
      # pool size h*w
      flag_integer("pool_size_h", hyper$pool_size_h1[i]),
      flag_integer("pool_size_w", hyper$pool_size_w1[i]),
      # fully-connected layer size
      flag_integer("fcl_size", hyper$fcl_size[i]),
      # fraction of the units to drop for the linear transformation of the inputs
      flag_numeric("dropout", hyper$dropout1[i]),
      # optimizer
      flag_string("optimizer", hyper$optimizer[i]),
      # learning rate
      flag_numeric("lr", hyper$learning_rate[i]),
      # training batch size
      flag_integer("batch_size", hyper$batch_size[i]),
      # num_epoch
      flag_integer("num_epoch", hyper$num_epoch[i]),
      # parameter to the early stopping callback
      flag_integer("patience", 15)
    )
  }else if (hyper$conv_layer=="2 layers"){
    FLAGS <- flags(
      flag_string("modality",hyper$modal[i]),
      flag_string("var_lst",hyper$var_lst[i]),
      
      flag_string("epoch",hyper$epoch_type[i]),
      # seed for train/test split and cnn
      flag_integer("seed",hyper$seed_split[i]),
      # m
      flag_integer("m", hyper$m[i]),
      # l
      flag_integer("l", hyper$l[i]),
      # stepsize
      flag_integer("step_size", hyper$step_size[i]),
      # hr indicator
      flag_boolean("hr", hyper$hr[i]),
      
      ##### layer1
      # kernel size: h*w
      flag_integer("kernel_size_h1", hyper$kernel_size_h1[i]),
      flag_integer("kernel_size_w1", hyper$kernel_size_w1[i]),
      # number of kernels
      flag_integer("num_kernel1", hyper$num_kernel1[i]),
      # pool size h*w
      flag_integer("pool_size_h1", hyper$pool_size_h1[i]),
      flag_integer("pool_size_w1", hyper$pool_size_w1[i]),
      # dropout: fraction of the units to drop for the linear transformation of the inputs
      flag_numeric("dropout1", hyper$dropout1[i]),
      
      ##### layer2
      # kernel size: h*w
      flag_integer("kernel_size_h2", hyper$kernel_size_h2[i]),
      flag_integer("kernel_size_w2", hyper$kernel_size_w2[i]),
      # number of kernels
      flag_integer("num_kernel2", hyper$num_kernel2[i]),
      # pool size h*w
      flag_integer("pool_size_h2", hyper$pool_size_h2[i]),
      flag_integer("pool_size_w2", hyper$pool_size_w2[i]),
      # dropout
      flag_numeric("dropout2", hyper$dropout2[i]),
      
      #### output layer
      # fully-connected layer size
      flag_integer("fcl_size", hyper$fcl_size[i]),
      # optimizer
      flag_string("optimizer", hyper$optimizer[i]),
      # learning rate
      flag_numeric("lr", hyper$learning_rate[i]),
      # training batch size
      flag_integer("batch_size", hyper$batch_size[i]),
      # num_epoch
      flag_integer("num_epoch", hyper$num_epoch[i]),
      # parameter to the early stopping callback
      flag_integer("patience", 15)
    )
  }else if (hyper$conv_layer=="3 layers"){
    FLAGS <- flags(
      flag_string("modality",hyper$modal[i]),
      flag_string("var_lst",hyper$var_lst[i]),
      
      flag_string("epoch",hyper$epoch_type[i]),
      # seed for train/test split and cnn
      flag_integer("seed",hyper$seed_split[i]),
      # m
      flag_integer("m", hyper$m[i]),
      # l
      flag_integer("l", hyper$l[i]),
      # stepsize
      flag_integer("step_size", hyper$step_size[i]),
      # hr indicator
      flag_boolean("hr", hyper$hr[i]),
      
      ##### layer1
      # kernel size: h*w
      flag_integer("kernel_size_h1", hyper$kernel_size_h1[i]),
      flag_integer("kernel_size_w1", hyper$kernel_size_w1[i]),
      # number of kernels
      flag_integer("num_kernel1", hyper$num_kernel1[i]),
      # pool size h*w
      flag_integer("pool_size_h1", hyper$pool_size_h1[i]),
      flag_integer("pool_size_w1", hyper$pool_size_w1[i]),
      # dropout: fraction of the units to drop for the linear transformation of the inputs
      flag_numeric("dropout1", hyper$dropout1[i]),
      
      ##### layer2
      # kernel size: h*w
      flag_integer("kernel_size_h2", hyper$kernel_size_h2[i]),
      flag_integer("kernel_size_w2", hyper$kernel_size_w2[i]),
      # number of kernels
      flag_integer("num_kernel2", hyper$num_kernel2[i]),
      # pool size h*w
      flag_integer("pool_size_h2", hyper$pool_size_h2[i]),
      flag_integer("pool_size_w2", hyper$pool_size_w2[i]),
      # dropout
      flag_numeric("dropout2", hyper$dropout2[i]),
      
      ##### layer3
      # kernel size: h*w
      flag_integer("kernel_size_h3", hyper$kernel_size_h3[i]),
      flag_integer("kernel_size_w3", hyper$kernel_size_w3[i]),
      # number of kernels
      flag_integer("num_kernel3", hyper$num_kernel3[i]),
      # pool size h*w
      flag_integer("pool_size_h3", hyper$pool_size_h3[i]),
      flag_integer("pool_size_w3", hyper$pool_size_w3[i]),
      # dropout
      flag_numeric("dropout3", hyper$dropout3[i]),
      
      #### output layer
      # fully-connected layer size
      flag_integer("fcl_size", hyper$fcl_size[i]),
      # optimizer
      flag_string("optimizer", hyper$optimizer[i]),
      # learning rate
      flag_numeric("lr", hyper$learning_rate[i]),
      # training batch size
      flag_integer("batch_size", hyper$batch_size[i]),
      # num_epoch
      flag_integer("num_epoch", hyper$num_epoch[i]),
      # parameter to the early stopping callback
      flag_integer("patience", 15)
    )
  }else{
    FLAGS <- flags(
      flag_string("modality",hyper$modal[i]),
      flag_string("var_lst",hyper$var_lst[i]),
      
      flag_string("epoch",hyper$epoch_type[i]),
      # seed for train/test split and cnn
      flag_integer("seed",hyper$seed_split[i]),
      # m
      flag_integer("m", hyper$m[i]),
      # l
      flag_integer("l", hyper$l[i]),
      # stepsize
      flag_integer("step_size", hyper$step_size[i]),
      # hr indicator
      flag_boolean("hr", hyper$hr[i]),
      
      ##### layer1
      # kernel size: h*w
      flag_integer("kernel_size_h1", hyper$kernel_size_h1[i]),
      flag_integer("kernel_size_w1", hyper$kernel_size_w1[i]),
      # number of kernels
      flag_integer("num_kernel1", hyper$num_kernel1[i]),
      # pool size h*w
      flag_integer("pool_size_h1", hyper$pool_size_h1[i]),
      flag_integer("pool_size_w1", hyper$pool_size_w1[i]),
      # dropout: fraction of the units to drop for the linear transformation of the inputs
      flag_numeric("dropout1", hyper$dropout1[i]),
      
      ##### layer2
      # kernel size: h*w
      flag_integer("kernel_size_h2", hyper$kernel_size_h2[i]),
      flag_integer("kernel_size_w2", hyper$kernel_size_w2[i]),
      # number of kernels
      flag_integer("num_kernel2", hyper$num_kernel2[i]),
      # pool size h*w
      flag_integer("pool_size_h2", hyper$pool_size_h2[i]),
      flag_integer("pool_size_w2", hyper$pool_size_w2[i]),
      # dropout
      flag_numeric("dropout2", hyper$dropout2[i]),
      
      ##### layer3
      # kernel size: h*w
      flag_integer("kernel_size_h3", hyper$kernel_size_h3[i]),
      flag_integer("kernel_size_w3", hyper$kernel_size_w3[i]),
      # number of kernels
      flag_integer("num_kernel3", hyper$num_kernel3[i]),
      # pool size h*w
      flag_integer("pool_size_h3", hyper$pool_size_h3[i]),
      flag_integer("pool_size_w3", hyper$pool_size_w3[i]),
      # dropout
      flag_numeric("dropout3", hyper$dropout3[i]),
      
      ##### layer4
      # kernel size: h*w
      flag_integer("kernel_size_h4", hyper$kernel_size_h4[i]),
      flag_integer("kernel_size_w4", hyper$kernel_size_w4[i]),
      # number of kernels
      flag_integer("num_kernel4", hyper$num_kernel4[i]),
      # pool size h*w
      flag_integer("pool_size_h4", hyper$pool_size_h4[i]),
      flag_integer("pool_size_w4", hyper$pool_size_w4[i]),
      # dropout
      flag_numeric("dropout4", hyper$dropout4[i]),
      
      #### output layer
      # fully-connected layer size
      flag_integer("fcl_size", hyper$fcl_size[i]),
      # optimizer
      flag_string("optimizer", hyper$optimizer[i]),
      # learning rate
      flag_numeric("lr", hyper$learning_rate[i]),
      # training batch size
      flag_integer("batch_size", hyper$batch_size[i]),
      # num_epoch
      flag_integer("num_epoch", hyper$num_epoch[i]),
      # parameter to the early stopping callback
      flag_integer("patience", 15)
    )
  }
}

########### function for CNN's majority vote and weighted accuracy
# y_pred has to be a vector.
majority_vote <- function(y_pred,w_size=1){    # window size = 2*w_size+1
  new_pred=y_pred
  for(i in c(1:length(y_pred))){
    new_pred[i]=round(mean(y_pred[max(0,i-w_size):min(i+w_size,length(y_pred))]))
  }
  return(new_pred)
}

weighted_acc <- function(y_pred,y_true){
  w1=sum(y_true)
  w0=length(y_true)-w1
  if (w1!=0){
    w_a=1/2*sum(y_pred[which(y_true==1)]==1)/w1+1/2*sum(y_pred[which(y_true==0)]==0)/w0
  }
  else w_a=mean(y_pred==y_true)
  return(w_a)
}

get_win_size = function(pred_train,cum_len,all_y,grid=1:50,idx_train_val){
  diff=rep(NA,length(grid))
  for (i in 1:length(grid)){
    before=NULL
    after=NULL
    for (j in 1:length(idx_train_val)){
      before=c(before,weighted_acc(pred_train[(cum_len[j]+1):cum_len[j+1]],all_y[(cum_len[j]+1):cum_len[j+1]]))
      mv_pred=majority_vote(pred_train[(cum_len[j]+1):cum_len[j+1]],w_size=grid[i])
      after=c(after,weighted_acc(mv_pred,all_y[(cum_len[j]+1):cum_len[j+1]]))
    }
    diff[i] = mean(after-before)
  }
  win_size=grid[which.max(diff)]
}

### find transition period
#sec_epoch_ty = sec[epoch_ty]
# when user tag sleep start, they usually haven't slept and still awake, so transition period may be a period (shorter) before and (longer) after the sleep-start tag
# when user tag awake (sleep ends), they usually have been awake for some time, so transition period may be a period (longer) before and (shorter) after the sleep-end tag.
get_transition = function(true_y,sec_epoch_ty,shorter_period = 30,longer_period = 60){  # shorter_period and longer_period measured in minutes.
  num_epoch_period = longer_period*60/sec_epoch_ty 
  num_epoch_period1 = shorter_period*60/sec_epoch_ty 
  
  df = tibble(true_y,lag1=lag(true_y),diff=true_y-lag1)
  
  # find change point
  start_sleep_idx = which(df$diff==1)
  start_awake_idx = which(df$diff==-1)
  # trans from awake to sleep, shorter period before and longer period after.
  trans_sleep_idx0 = start_sleep_idx - num_epoch_period1   # in total 18 epochs (90 mins) in each transition period for 5min.
  trans_sleep_idx = start_sleep_idx + num_epoch_period - 1
  # trans from sleep to awake, longer period before and shorter period after.
  trans_awake_idx = start_awake_idx - num_epoch_period + 1
  trans_awake_idx1 = start_awake_idx + num_epoch_period1
  
  transition = rep(0,nrow(df))
  for (k in 1:length(start_sleep_idx)){
    transition[(max(1,trans_sleep_idx0[k])):(min(trans_sleep_idx[k],nrow(df)))] = 1
  }
  for (k in 1:length(start_awake_idx)){
    transition[(max(1,trans_awake_idx[k])):(min(trans_awake_idx1[k],nrow(df)))] = 1
  }
  return(transition)
}




#### compute metrics
compute_metrics = function(pred_binary,test_y){
  
  #conf_matrix <- table(pred_binary, test_y) # create confusion matrix pre majority vote
  conf_mat_manual =  matrix(c(sum(test_y==0&pred_binary==0),
                              sum(test_y==1&pred_binary==0),
                              sum(test_y==0&pred_binary==1),
                              sum(test_y==1&pred_binary==1)),2,2)
  colnames(conf_mat_manual) = paste0('pred',0:1)
  rownames(conf_mat_manual) = paste0('true',0:1)
  # AUC
  AUC = DescTools::AUC(pred_binary,test_y)
  
  # accuracy
  Accuracy =mean(pred_binary==test_y)
  
  # specificity = TN/(TN+FP)
  Specificity = conf_mat_manual[1,1]/sum(conf_mat_manual[1,])
  #specificity(conf_matrix, negative = "0")
  
  # precision = TP/(TP+FP)
  Precision = conf_mat_manual[2,2]/sum(conf_mat_manual[,2])
  #precision(conf_matrix, relevant = "1")
  
  # recall = TP/(TP+FN)
  Recall = conf_mat_manual[2,2]/sum(conf_mat_manual[2,])
  #recall(conf_matrix, relevant = "1")
  
  # F1 score = 2 * (Precision * Recall) / (Precision + Recall)
  F1score = 2 * (Precision * Recall) / (Precision + Recall)
  
  # weighted accuracy
  wACC = 1/2*Specificity + 1/2*Recall
  
  return(list(conf_mat=conf_mat_manual, AUC=AUC,
              Accuracy=Accuracy,wACC=wACC,Specificity=Specificity,
              Precision=Precision,Recall=Recall,F1score=F1score))
}