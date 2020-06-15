#oof_lgb,oof_xgb为用来stacking的模型结果，其中clf_3为stacking的模型，可以选择lr，贝叶斯的简单的模型效果较好。
def Basyen_stacking(oof_lgb,oof_xgb,predictions_lgb,predictions_xgb,sub,target):
    res = sub.copy()
    train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2000)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], target[trn_idx]
        val_data, val_y = train_stack[val_idx], target[val_idx]
        
        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)
        
        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10
    
    cross_validation_loss = mean_squared_error(target, oof_stack)
    print(cross_validation_loss)

    res[1] = predictions
    mean = res[1].mean()
    print('mean:',mean)
    res.to_csv("./Basyen_stacking.csv",index=False,header = None)
    return cross_validation_loss
 