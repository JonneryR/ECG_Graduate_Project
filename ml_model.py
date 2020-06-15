
###linear regression,tree regression,svm,knn,rf,adaboost,gdbt，所有的sklearn的简单模型在这里都可以使用
def sklearn_seed_cv(model,name,train,label,test,test_id,seed = 2018,round = 5000,n_folds = 5):

    res = test_id.copy()
    oof = np.zeros(train.shape[0])
    predictions = np.zeros(test.shape[0])

    sk_model = model 
    
    train = pd.DataFrame(train,dtype=np.float).fillna(-1).values
    test = pd.DataFrame(test,dtype = np.float).fillna(-1).values
    for data in [train,test]:
        where_are_nan = np.isnan(data)
        where_are_inf = np.isinf(data)
        data[where_are_nan] = -1
        data[where_are_inf] = -2
    gc.collect()
    

    print('Training set and valid set has been splited completely!：',train.shape,test.shape) 
    
    skf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    baseloss = []  
    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        sk_model.fit(train[train_index], label[train_index])
        gc.collect()

        oof[test_index] = sk_model.predict(train[test_index])
        valid_loss = mean_squared_error(label[test_index],oof[test_index])
        baseloss.append(valid_loss)
        predictions += sk_model.predict(test)/5
        gc.collect()
    print('MSE:', baseloss, np.mean(baseloss))

    # 加权平均
    res[1] = predictions
    mean = res[1].mean()
    print('mean:',mean)
    res.to_csv("./submit/%s_baseline_%.6f.csv"%(name,np.mean(baseloss)), index=False,header = None)

    return res,oof
def lgb_seed_cv(train_csr,label,predict_csr,test,round = 3000,seed = 2020,n_folds = 5):
    
    # X_loc_test = lgb.Dataset(base_predict_csr)
    res = test.copy()
    oof = np.zeros(train_csr.shape[0])
    predictions = np.zeros(predict_csr.shape[0])
    '''
    train_csr = pd.DataFrame(train_csr,dtype=np.float).fillna(-1).values
    predict_csr = pd.DataFrame(predict_csr,dtype = np.float).fillna(-1).values
    for data in [train,test]:
        where_are_nan = np.isnan(data)
        where_are_inf = np.isinf(data)
        data[where_are_nan] = -1
        data[where_are_inf] = -2
    '''
    print('Training set and valid set has been splited completely!：',train_csr.shape,predict_csr.shape) 
    lgb_model = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=120, reg_alpha=0.1, reg_lambda=5, max_depth=-1,
                                    n_estimators=round, subsample=0.9, colsample_bytree=0.77, 
                                    subsample_freq=1, learning_rate=0.05,random_state=1000, n_jobs=16, 
                                    min_child_weight=4, min_child_samples=30, min_split_gain=0)
    
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    baseloss = []
    loss = 0
    for index,(train_index, test_index) in enumerate(folds.split(train_csr,label)):
        print("Fold", index)
        
        lgb_model.fit(train_csr.iloc[train_index],label[train_index], 
                      eval_set=[(train_csr.iloc[train_index],label[train_index]),
                                (train_csr.iloc[test_index],label[test_index])],
                                verbose=50, early_stopping_rounds=100)

        oof[test_index] = lgb_model.predict(train_csr.iloc[test_index], num_iteration=lgb_model.best_iteration_)

        valid_loss = mean_squared_error(label[test_index],oof[test_index])
        baseloss.append(valid_loss)
        predictions += lgb_model.predict(predict_csr, num_iteration=lgb_model.best_iteration_)
        gc.collect()
    print('MSE:', baseloss, np.mean(baseloss))

    # 加权平均
    res[1] = predictions/5
    mean = res[1].mean()
    print('mean:',mean)
    res.to_csv("./submit/lgb_baseline_%.6f.csv"%np.mean(baseloss), index=False,header = None)

    return res,oof
 

def xgb_seed_cv(train_csr,label,predict_csr,test,round = 3000,seed = 2020,n_folds = 5):
    
    # X_loc_test = lgb.Dataset(base_predict_csr)
    res = test.copy()
    oof = np.zeros(len(train_csr))
    predictions = np.zeros(len(predict_csr))
    '''
    for data in [train,test]:
        where_are_nan = np.isnan(data)
        where_are_inf = np.isinf(data)
        data[where_are_nan] = -1
        data[where_are_inf] = -2
    '''
    print('Training set and valid set has been splited completely!：',train_csr.shape,predict_csr.shape) 
    xgb_model = xgb.XGBRegressor(boosting_type='gbdt', num_leaves=48, max_depth=8, 
                                  learning_rate=0.05, n_estimators=round,subsample=0.8,
                                  colsample_bytree=0.6, reg_alpha=3, reg_lambda=5, 
                                  seed=1000, nthread=10,verbose=50)
    
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    baseloss = []
    loss = 0
    for index,(train_index, test_index) in enumerate(folds.split(train_csr,label)):
        print("Fold", index)
        
        xgb_model.fit(train_csr[train_index],label[train_index], 
                      eval_set=[(train_csr[train_index],label[train_index]),
                                (train_csr[test_index],label[test_index])],eval_metric = 'mse',
                                verbose=50, early_stopping_rounds=100)

        oof[test_index] = xgb_model.predict(train_csr[test_index], ntree_limit=xgb_model.best_iteration)

        valid_loss = mean_squared_error(label[test_index],oof[test_index])
        baseloss.append(valid_loss)
        predictions += xgb_model.predict(predict_csr, ntree_limit=xgb_model.best_iteration)
        
        gc.collect()
    print('MSE:', baseloss, np.mean(baseloss))

    # 加权平均
    res[1] = predictions/5
    mean = res[1].mean()
    print('mean:',mean)
    res.to_csv("./submit/xgb_baseline_%.6f.csv"%np.mean(baseloss), index=False,header = None)

    return res,oof
 