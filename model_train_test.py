



'''Main code for training and test the models. For different outcomes (hospitalization/ICU/intubation/mortality), code may slightly change'''


is_BWH=False#True#
col_y='flag_ICU'#'flag_ventilator'#
#cols_rep=['AUC', 'micro F1-score', 'weighted F1-score','tr AUC',  'tr micro F1-score','tr weighted F1-score']
cols_rep=['AUC', 'weighted F1-score']


#---------------------------- Functions


import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn import preprocessing
from scipy import stats

def chi2_cols(y,x):
    '''
    input:
    y: 1-d binary label array
    x: 1-d binary feature array
    
    return:
    chi2 statistic and p-value
    '''
    y_list=y.astype(int).tolist()
    x_list=x.astype(int).tolist()
    freq=np.zeros([2,2])

    for i in range(len(y_list)):
        if y_list[i]==0 and x_list[i]==0:
          freq[0,0]+=1
        if y_list[i]==1 and x_list[i]==0:
          freq[1,0]+=1
        if y_list[i]==0 and x_list[i]==1:
          freq[0,1]+=1
        if y_list[i]==1 and x_list[i]==1:
          freq[1,1]+=1
    y_0_sum=np.sum(freq[0,:])
    y_1_sum=np.sum(freq[1,:])
    x_0_sum=np.sum(freq[:,0])
    x_1_sum=np.sum(freq[:,1])
    total=y_0_sum+y_1_sum
    y_0_ratio=y_0_sum/total
    freq_=np.zeros([2,2])    
    freq_[0,0]=x_0_sum*y_0_ratio
    freq_[0,1]=x_1_sum*y_0_ratio
    freq_[1,0]=x_0_sum-freq_[0,0]
    freq_[1,1]=x_1_sum-freq_[0,1]

    stat,p_value=stats.chisquare(freq,freq_,axis=None)    
    return p_value#stat,
def stat_test(df, y):
    name = pd.DataFrame(df.columns,columns=['Variable'])
    df0=df[y==0]
    df1=df[y==1]
    pvalue=[]
    y_corr=[]
    for col in df.columns:
        if df[col].nunique()==2:
            pvalue.append(chi2_cols( y,df[col]))
            # pvalue.append(stats.ttest_ind(df0[col], df1[col], equal_var = False,nan_policy='omit').pvalue)
        else:
            pvalue.append(stats.ks_2samp(df0[col], df1[col]).pvalue)
        y_corr.append(df[col].corr(y))
    name['All_mean']=df.mean().values
    name['y1_mean']=df1.mean().values
    name['y0_mean']=df0.mean().values
    name['All_std']=df.std().values
    name['y1_std']=df1.std().values
    name['y0_std']=df0.std().values
    name['p-value']=pvalue
    name['y_corr']=y_corr
    return name.sort_values(by=['p-value'])#[['Variable','p-value','y_corr']]
    #name.sort_values(by=['p-value']).drop(['All_std','y1_std','y0_std','p-value'],axis=1)

def high_corr(df, thres=0.8):
    corr_matrix_raw = df.corr()
    corr_matrix = corr_matrix_raw.abs()
    high_corr_var_=np.where(corr_matrix>thres)
    high_corr_var=[(corr_matrix.index[x],corr_matrix.columns[y], corr_matrix_raw.iloc[x,y]) for x,y in zip(*high_corr_var_) if x!=y and x<y]
    return high_corr_var

def df_fillna(df):
    df_nullsum=df.isnull().sum()
    for col in df_nullsum[df_nullsum>0].index:
        # df[col+'_isnull']=df[col].isnull().values
        df[col]=df[col].fillna(df[col].median())    
    return df
def df_drop(df_new, drop_cols):
    return df_new.drop(df_new.columns[df_new.columns.isin(drop_cols)], axis=1)

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,f1_score, roc_curve, auc, accuracy_score     
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE#,RFECV # 
import lightgbm as lgb   

def clf_F1(best_C_grid, best_F1, best_F1std, classifier, X_train, y_train,C_grid,nFolds, silent=True,seed=2020,scoring='f1'):#
    # global best_C_grid,best_F1, best_F1std
    results= cross_val_score(classifier, X_train, y_train, cv=StratifiedKFold(n_splits=nFolds,shuffle=True,random_state=seed), n_jobs=2,scoring=scoring)#cross_validation.
    F1, F1std = results.mean(), results.std()
    if silent==False:
        print(C_grid,F1, F1std)        
    if F1>best_F1:
        best_C_grid=C_grid
        best_F1, best_F1std=F1, F1std
    return best_C_grid, best_F1, best_F1std

def my_RFE(df_new, col_y='flag_ICU', my_range=range(1,11,1), my_penalty='l1', my_C = 0.1, cvFolds=5,step=1,scoring='f1'):
    F1_all_rfe=[]
    Xraw=df_new.drop(col_y, axis=1).values#
    y= df_new[col_y].values
    names=df_new.drop(col_y, axis=1).columns
    for n_select in my_range:
        scaler = preprocessing.StandardScaler()#MinMaxScaler
        X = scaler.fit_transform(Xraw)
        clf=LogisticRegression(C=my_C,penalty=my_penalty,class_weight= 'balanced',solver='liblinear')#tol=0.01,
        # clf = LinearSVC(penalty='l1',C=0.1,class_weight= 'balanced', dual=False)
        rfe = RFE(clf, n_select, step=step)
        rfe.fit(X, y.ravel())
        X = df_new.drop(col_y, axis=1).drop(names[rfe.ranking_>1], axis=1).values
    #     # id_keep_1st=df_new.drop(col_y, axis=1).drop(drop_col, axis=1).columns
    #     id_keep_1st= names[rfe.ranking_==1].values    
        X = scaler.fit_transform(X)
        # clf = LogisticRegressionCV(Cs=[10**-1,10**0, 10], penalty='l1',solver='liblinear', cv=5,scoring='f1', random_state=2020)#, tol=0.01
        # clf.fit(X, y)
        # clf.scores_
        best_F1, best_F1std=0.1, 0
        best_C_grid=0
        for C_grid in [0.01,0.1,1,10]:
            clf=LogisticRegression(C=C_grid,class_weight= 'balanced',solver='liblinear')#penalty=my_penalty,
            best_C_grid, best_F1, best_F1std=clf_F1(best_C_grid, best_F1, best_F1std,clf,X, y,C_grid,cvFolds,scoring=scoring)
        F1_all_rfe.append((n_select, best_F1, best_F1std))
    F1_all_rfe=pd.DataFrame(F1_all_rfe, index=my_range,columns=['n_select',"best_score_mean","best_score_std"])
    F1_all_rfe['best_score_']= F1_all_rfe['best_score_mean']#-F1_all_rfe['best_score_std']
    ######
    ######
    X = scaler.fit_transform(Xraw)
    clf=LogisticRegression(C=my_C,penalty=my_penalty,class_weight= 'balanced',solver='liblinear')#0.
    rfe = RFE(clf, F1_all_rfe.loc[F1_all_rfe['best_score_'].idxmax(),'n_select'], step=step)
    rfe.fit(X, y.ravel())
    id_keep_1st= names[rfe.ranking_==1].values
    return id_keep_1st, F1_all_rfe



def my_train(X_train, y_train, model='LR', penalty='l1', cv=5, scoring='f1', class_weight= 'balanced',seed=2020):    
    if model=='SVM':
        svc=LinearSVC(penalty=penalty, class_weight= class_weight, dual=False, max_iter=5000)#, tol=0.0001
        param_grid = {'C':[0.01,0.1,1,10,100]} #'kernel':('linear', 'rbf'), 
        gsearch = GridSearchCV(svc, param_grid, cv=cv, scoring=scoring) 
    elif model=='LGB':        
        param_grid = {
            'num_leaves': range(2,7,1),
            'n_estimators': range(40,100,10),
            'colsample_bytree':[0.05, 0.075,  0.1,  0.15,0.2,0.3]# [0.6, 0.7, 0.8, 0.9]#[0.6, 0.75, 0.9]
            # 'reg_alpha': [0.1, 0.5],# 'min_data_in_leaf': [30, 50, 100, 300, 400],
            # 'lambda_l1': [0, 1, 1.5],# 'lambda_l2': [0, 1]
            }
        lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', learning_rate=0.1, class_weight= class_weight, random_state=seed)# eval_metric='auc' num_boost_round=2000,
        gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=cv,n_jobs=2, scoring=scoring)
    elif model=='RF': 
        rfc=RandomForestClassifier(n_estimators=100, random_state=seed, class_weight= class_weight, n_jobs=2)
        param_grid = { 
            'max_features':[0.05, 0.1, 0.2, 0.3],#, 0.4, 0.5, 0.6, 0.7, 0.8 [ 'sqrt', 'log2',15],#'auto'  1.0/3,
            'max_depth' : range(2,6,1)#[2, 10]
    #     'criterion' :['gini', 'entropy'] #min_samples_split = 10, 
        }
        gsearch = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cv, scoring=scoring)
    else:
        LR = LogisticRegression(penalty=penalty, class_weight= class_weight,solver='liblinear', random_state=seed)
        parameters = {'C':[0.1,1,10] } 
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring) 
        # clf = LogisticRegressionCV(Cs=[10**-1,10**0, 10], penalty=penalty, class_weight= class_weight,solver='liblinear', cv=cv, scoring=scoring, random_state=seed)#, tol=0.01
    gsearch.fit(X_train, y_train)
    clf=gsearch.best_estimator_
    if model=='LGB' or model=='RF': 
        print('Best parameters found by grid search are:', gsearch.best_params_)
    # print('train set accuracy:', clf.score(X_train, y_train))
    # print('train set accuracy:', gsearch.best_score_)
    return clf

def cal_f1_scores(y, y_pred_score):
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    thresholds = sorted(set(thresholds))#
    metrics_all = []
    for thresh in thresholds:
      y_pred = np.array((y_pred_score > thresh))
      metrics_all.append(( thresh,auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'),f1_score(y, y_pred, average='weighted')))
    metrics_df = pd.DataFrame(metrics_all, columns=['thresh','tr AUC',  'tr micro F1-score', 'tr macro F1-score','tr weighted F1-score'])
    return metrics_df.sort_values(by = 'tr weighted F1-score', ascending = False).head(1)#['thresh'].values[0]
def cal_f1_scores_te(y, y_pred_score,thresh):
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    y_pred = np.array((y_pred_score > thresh))
    metrics_all = [ (thresh,auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'),f1_score(y, y_pred, average='weighted'))]
    metrics_df = pd.DataFrame(metrics_all, columns=['thresh','AUC',  'micro F1-score', 'macro F1-score','weighted F1-score'])
    return metrics_df

def my_test(X_train, xtest, y_train, ytest, clf, target_names, report=False, model='LR'): 
    if model=='SVM': 
        ytrain_pred_score=clf.decision_function(X_train)
    else:
        ytrain_pred_score=clf.predict_proba(X_train)[:,1]
    metrics_tr =cal_f1_scores( y_train, ytrain_pred_score)
    thres_opt=metrics_tr['thresh'].values[0]   
    # ytest_pred=clf.predict(xtest)
    if model=='SVM': 
        ytest_pred_score=clf.decision_function(xtest)
    else:
        ytest_pred_score=clf.predict_proba(xtest)[:,1]
    # fpr, tpr, thresholds = roc_curve(ytest, ytest_pred_score)
    # if report:
    #     print(classification_report(ytest, ytest_pred, target_names=target_names,digits=4))
    metrics_te = cal_f1_scores_te(ytest, ytest_pred_score,thres_opt)    
    return metrics_te.merge(metrics_tr,on='thresh'),thres_opt,ytest_pred_score
    # (f1_score(ytest, ytest_pred), auc(fpr, tpr), f1_score(ytest, ytest_pred, average='micro'), f1_score(ytest, ytest_pred, average='macro'),f1_score(ytest, ytest_pred, average='weighted'))#,accuracy_score(ytest, ytest_pred)
# 

def tr_predict(df_new, col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring='f1', test_size=0.2,report=False, RFE=False,pred_score=False):
    scaler = preprocessing.StandardScaler()#MinMaxScaler           
    y= df_new[col_y].values
    metrics_all=[]
    if is_BWH:
      my_seeds=range(2020, 2021)
      # DATA_PATH = 'path/'
      # hos_stat_latest = pd.read_csv(DATA_PATH + 'data.csv')#,index_col=0
      mask=df_new.index.isin(hos_stat_latest.loc[(hos_stat_latest['Hospital']!='BWH'),'PID'].values)
    else:
      my_seeds=range(2020, 2025)
    for seed in my_seeds:
        X = df_new.drop([col_y, 'vitals_lstm']+['vitals_lstm_'+str(i) for i in range(2020,2025) if i!=seed], axis=1).values
        name_cols=df_new.drop([col_y, 'vitals_lstm']+['vitals_lstm_'+str(i) for i in range(2020,2025) if i!=seed], axis=1).columns.values 
        X = scaler.fit_transform(X)
        if is_BWH:
          X_train, xtest, y_train, ytest = X[mask,:],X[~mask,:],y[mask],y[~mask]
        else:
          X_train, xtest, y_train, ytest = train_test_split(X, y, stratify=y, test_size=test_size,  random_state=seed)#
          #if seed==2021:
            #print(ytest)
        
        if RFE:
            df_train=pd.DataFrame(X_train, columns=name_cols )
            df_train[col_y]=y_train
            id_keep_1st, F1_all_rfe=my_RFE(df_train, col_y=col_y, cvFolds=cv_folds, scoring=scoring)# my_penalty='l1', my_C = 1, my_range=range(25,46,5), 
            print(F1_all_rfe)
            print(list(id_keep_1st))
            X_train=df_train[id_keep_1st]
            df_test=pd.DataFrame(xtest, columns=name_cols )
            xtest=df_test[id_keep_1st]
            name_cols=id_keep_1st
        clf = my_train(X_train, y_train, model=model, penalty=penalty, cv=cv_folds, scoring=scoring, class_weight= 'balanced',seed=seed)    
        metrics_te,thres_opt, ytest_pred_score=my_test(X_train, xtest, y_train, ytest, clf, target_names, report=report, model=model)
        metrics_all.append(metrics_te)
    metrics_df=pd.concat(metrics_all)
    metrics_df = metrics_df[cols_rep].describe().T[['mean','std']].stack().to_frame().T

    if pred_score and is_BWH:
        ytest_pred=df_new.loc[~mask,[col_y]].copy()
        print(ytest_pred.shape,ytest_pred.head())
        print('thres_opt',thres_opt)
        ytest_pred['ytest_pred_score']=ytest_pred_score
        ytest_pred['ytest_pred']=ytest_pred_score>thres_opt
        ytest_pred['ytest']=ytest
        # refit using all samples to get non-biased coef.
    
    #-------- make sure that fitting all dataset using 'vitals_lstm' column
    X = df_new.drop([col_y]+['vitals_lstm_'+str(i) for i in range(2020,2025)], axis=1).values
    name_cols=df_new.drop([col_y]+['vitals_lstm_'+str(i) for i in range(2020,2025)], axis=1).columns.values
    X = scaler.fit_transform(X)
    
    
    clf.fit(X, y)
    if pred_score and (not is_BWH):
        print('predict 17 additional patients')
        df17 = pd.read_csv('....csv')
        df17.columns =['ID', 'LDH', 'CRP (mg/L)', 'Sodium', 'Calcium', 'Anion Gap',
                       'medication_Insulin_related', 'SpO2_percentage', 'radiology_Opacity','hours_from_adm_to_icu', 'mech_vent']
        df17 = df17[name_cols]
        from sklearn.impute import SimpleImputer
        imp_mean = SimpleImputer(missing_values=-99, strategy='median')
        imp_mean.fit(df_new.drop([col_y], axis=1))
        xtest17=imp_mean.transform(df17)
        xtest17 = scaler.transform(xtest17)
        if model=='SVM': 
            ytest_pred_score=clf.decision_function(xtest17)
            y_all_pred_score=clf.decision_function(X)
        else:
            ytest_pred_score=clf.predict_proba(xtest17)[:,1]
            y_all_pred_score=clf.predict_proba(X)[:,1]
        df17['ytest_pred_score']=ytest_pred_score
        ytest_pred=df17#[['ID','ytest_pred_score']]#,'hours_from_adm_to_icu', 'mech_vent'
        thres_opt=cal_f1_scores(y, y_all_pred_score)['thresh'].values[0] 
        print('thres_opt',thres_opt)
        
    #print(name_cols)
    #print(np.round(clf.coef_[0],2))
    #print(list(zip(name_cols, np.round(clf.coef_[0],2))))
    
    if model=='LGB' or model=='RF': 
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.feature_importances_,2))),columns=['Variable','coef_'])
    else:
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.coef_[0],2))),columns=['Variable','coef_'])
        df_coef_= df_coef_.append({'Variable': 'intercept_','coef_': np.round(clf.intercept_,2)}, ignore_index=True)
    df_coef_['coef_abs']=df_coef_['coef_'].abs()
    if pred_score:# and is_BWH:#==True
      return df_coef_.sort_values('coef_abs', ascending=False)[['Variable','coef_']], metrics_df,ytest_pred
    else:
      return df_coef_.sort_values('coef_abs', ascending=False)[['Variable','coef_']], metrics_df



def tr_predict1(df_new, col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring='f1', test_size=0.2,report=False, RFE=False,pred_score=False):
    scaler = preprocessing.StandardScaler()#MinMaxScaler           
    y= df_new[col_y].values
    metrics_all=[]
    if is_BWH:
      my_seeds=range(2020, 2021)
      mask=df_new.index.isin(hos_stat_latest.loc[(hos_stat_latest['Hospital']!='BWH'),'PID'].values)
    else:
      my_seeds=range(2020, 2025)
    for seed in my_seeds:
        X = df_new.drop([col_y], axis=1).values
        name_cols=df_new.drop([col_y], axis=1).columns.values 
        X = scaler.fit_transform(X)
        if is_BWH:
          X_train, xtest, y_train, ytest = X[mask,:],X[~mask,:],y[mask],y[~mask]
        else:
          X_train, xtest, y_train, ytest = train_test_split(X, y, stratify=y, test_size=test_size,  random_state=seed)#
          #if seed==2021:
            #print(ytest)
        
        if RFE:
            df_train=pd.DataFrame(X_train, columns=name_cols )
            df_train[col_y]=y_train
            id_keep_1st, F1_all_rfe=my_RFE(df_train, col_y=col_y, cvFolds=cv_folds, scoring=scoring)# my_penalty='l1', my_C = 1, my_range=range(25,46,5), 
            print(F1_all_rfe)
            print(list(id_keep_1st))
            X_train=df_train[id_keep_1st]
            df_test=pd.DataFrame(xtest, columns=name_cols )
            xtest=df_test[id_keep_1st]
            name_cols=id_keep_1st
        clf = my_train(X_train, y_train, model=model, penalty=penalty, cv=cv_folds, scoring=scoring, class_weight= 'balanced',seed=seed)    
        metrics_te,thres_opt, ytest_pred_score=my_test(X_train, xtest, y_train, ytest, clf, target_names, report=report, model=model)
        metrics_all.append(metrics_te)
    metrics_df=pd.concat(metrics_all)
    metrics_df = metrics_df[cols_rep].describe().T[['mean','std']].stack().to_frame().T

    if pred_score and is_BWH:
        ytest_pred=df_new.loc[~mask,[col_y]].copy()
        print(ytest_pred.shape,ytest_pred.head())
        print('thres_opt',thres_opt)
        ytest_pred['ytest_pred_score']=ytest_pred_score
        ytest_pred['ytest_pred']=ytest_pred_score>thres_opt
        ytest_pred['ytest']=ytest
        # refit using all samples to get non-biased coef.
    clf.fit(X, y)
    if pred_score and (not is_BWH):
        print('predict 17 additional patients')
        df17 = pd.read_csv('....csv')
        df17.columns =['ID', 'LDH', 'CRP (mg/L)', 'Sodium', 'Calcium', 'Anion Gap',
                       'medication_Insulin_related', 'SpO2_percentage', 'radiology_Opacity','hours_from_adm_to_icu', 'mech_vent']
        df17 = df17[name_cols]
        from sklearn.impute import SimpleImputer
        imp_mean = SimpleImputer(missing_values=-99, strategy='median')
        imp_mean.fit(df_new.drop([col_y], axis=1))
        xtest17=imp_mean.transform(df17)
        xtest17 = scaler.transform(xtest17)
        if model=='SVM': 
            ytest_pred_score=clf.decision_function(xtest17)
            y_all_pred_score=clf.decision_function(X)
        else:
            ytest_pred_score=clf.predict_proba(xtest17)[:,1]
            y_all_pred_score=clf.predict_proba(X)[:,1]
        df17['ytest_pred_score']=ytest_pred_score
        ytest_pred=df17#[['ID','ytest_pred_score']]#,'hours_from_adm_to_icu', 'mech_vent'
        thres_opt=cal_f1_scores(y, y_all_pred_score)['thresh'].values[0] 
        print('thres_opt',thres_opt)
    if model=='LGB' or model=='RF': 
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.feature_importances_,2))),columns=['Variable','coef_'])
    else:
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.coef_[0],2))),columns=['Variable','coef_'])
        df_coef_= df_coef_.append({'Variable': 'intercept_','coef_': np.round(clf.intercept_,2)}, ignore_index=True)
    df_coef_['coef_abs']=df_coef_['coef_'].abs()
    if pred_score:# and is_BWH:#==True
      return df_coef_.sort_values('coef_abs', ascending=False)[['Variable','coef_']], metrics_df,ytest_pred
    else:
      return df_coef_.sort_values('coef_abs', ascending=False)[['Variable','coef_']], metrics_df


def get_odds_ratio(df, col_y = 'flag_ventilator'):

  import statsmodels.api as sm
  import pylab as pl

  X=df.drop(columns=[col_y])
  Y=df[col_y]

  # logit = sm.Logit(Y, X)
  logit = sm.Logit(Y, sm.add_constant(X))

  # fit the model
  result = logit.fit_regularized()
  print(result.summary())

  # odds ratios and 95% CI + Coef
  Coef_CI = pd.concat([result.params, np.exp(result.params), result.pvalues, np.exp(result.conf_int()).astype(float),  result.conf_int()], axis=1)
  Coef_CI.columns = ['Coef_Binary', 'Odds_Ratio','P_Value_Coef', 'Odds_Ratio_2.5%', 'Odds_Ratio_97.5%',  'Coef_Binary_2.5%', 'Coef_Binary_97.5%']
  print(Coef_CI)
  Coef_CI['Abs_Coef_Binary'] = Coef_CI['Coef_Binary'].abs()

  return Coef_CI#.sort_values(['Abs_Coef_Binary'], ascending = False).drop(['Abs_Coef_Binary'], axis = 1) #= get_odds_ratio(df, col_y = 'flag_ventilator')








#---------------------------- load


import os
DATA_PATH = 'data_path/'
os.listdir(DATA_PATH)

df = pd.read_csv(DATA_PATH + 'data_file.csv',index_col=0)

df_ori=df.copy()

# adding census
df_census = pd.read_csv('census_file.csv')
df_census=df_census[['PID','Total Non-COVID','Total COVID','Total Elective Surgery']]
df=df.merge(df_census,how='left', on='PID')
df=df_drop(df, ['BMC_TRICYCLICS (SERUM)_NEG_1tau'])

df['primary_race_Unknown']=df['primary_race_Unknown']+df['primary_race_Hispanic or Latino']
df=df_drop(df, ['primary_race_Hispanic or Latino'])

df.head()

print(df.shape)




#---------------------------- preprocessing


# drop as required

df.index=df['PID']

import pickle
data2=pickle.load(open('med_file.pkl','rb'))
med_list=list(data2[1].keys())
drop_cols=med_list
drop_cols.append('sym_Cough')
drop_cols+=['sym_Fever','sym_Cough','sym_Dyspnea','sym_Fatigue','sym_Diarrhea','sym_Nausea','sym_Vomiting','sym_Abdominal_pain','sym_Loss_of_smell','sym_Loss_of_taste','sym_Chest_pain','sym_Headache','sym_Sore_throat','sym_Hemoptysis','sym_Myalgia'
]

df=df_drop(df, drop_cols)
print(df.shape)


drop_cols=['PID']

if col_y==  'flag_ICU':
    drop_cols+=['flag_ventilator']
if col_y==  'flag_ventilator':
    drop_cols+=['flag_ICU']
df=df_drop(df, drop_cols)

print(df.shape)

df.dtypes.unique()

df_new=df.copy()
df_new.shape

hc=high_corr(df_new, thres=0.8)
print(hc)
drop_cols=[i[0] for i in hc if i[0] not in ['flag_ICU','vitals_lstm','vitals_lstm_2020','vitals_lstm_2021','vitals_lstm_2022','vitals_lstm_2023','vitals_lstm_2024']]
df_new=df_drop(df_new, drop_cols)

df_new.shape

df_new_std=df_new.std()
drop_cols=list(df_new_std[df_new_std<0.05].index)
df_new=df_drop(df_new, drop_cols)
#print(high_corr(df_new, thres=0.8))
df_new.shape


result=stat_test(df_new, df_new[col_y])
result



# AUC or F1

my_scoring='roc_auc'



#---------------------------- before stat select


from warnings import filterwarnings
filterwarnings('ignore')


df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=metrics_df.rename(index={0: 'LR-L1'})# df_AUCs=pd.concat([df_AUCs,df_AUC.rename(index={0: 'LR-L1'})])
# # df_coef_[df_coef_['coef_']!=0]

df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='SVM',penalty='l1',cv_folds=5,scoring=my_scoring)
# metrics_df.describe()
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'SVM-L1'})])

df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='LGB',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'GBT'})])
# df_coef_.merge(result, on='Variable').merge(df_count,how='left', on='Variable').fillna(len(df))[['Variable','coef_','y_corr','p-value','count','y1_mean', 'y0_mean']]

df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='RF',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'RF'})])
# df_coef_.merge(result, on='Variable').merge(df_count,how='left', on='Variable').fillna(len(df))[['Variable','coef_','y_corr','p-value','count','y1_mean', 'y0_mean']]
df_AUCs



#---------------------------- after stat select


drop_cols=result.loc[result['p-value']>0.05,'Variable'].values
df_new=df_drop(df_new, drop_cols)
df_new.shape


df_new.columns
df_new.shape

drop_cols=[]
for col in df_new.columns:
    if (df_new[col].nunique()==2) & (df_new[col].std()<0.05):
        drop_cols.append(col)
        # print(col, df_new[col].std())
df_new=df_drop(df_new, drop_cols)
df_new.shape


df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring)
if 'df_AUCs' in globals():
  df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'LR-L1_'})])
else:
  df_AUCs=metrics_df.rename(index={0: 'LR-L1_'})
#df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'LR-L1_'})])

df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='SVM',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'SVM-L1_'})])

df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='LGB',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'GBT_'})])
 
df_coef_,metrics_df = tr_predict(df_new, col_y=col_y,target_names = ['0', '1'], model='RF',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'RF_'})])
df_AUCs



#---------------------------- TOP 10 + L1LR


# BMC protocol


df_prot=df_ori[['flag_ICU','vitals_lstm_2020','vitals_lstm_2021','vitals_lstm_2022','vitals_lstm_2023','vitals_lstm_2024','vitals_lstm']]
prot_labs=['BMC_HEMATOCRIT',
 'BMC_HEP B SURFACE AB',
 'BMC_HEP B SURFACE AG_PRELIMINARY RESULT: REACTIVE. REFER TO CONFIRMATORY TEST.',
 'BMC_HEP B SURFACE AG_NON-REACTIVE',
 'BMC_HEPATITIS B CORE AB_REACTIVE',
 'BMC_HEPATITIS B CORE AB_NON-REACTIVE',
 'merged_RED_BLOOD_CELL_COUNT',
 'merged_HEMOGLOBIN_BLOOD',
 'merged_MEAN_CORPUSCULAR_VOLUME',
 'merged_MEAN_CORPUSCULAR_HEMOGLOBIN',
 'merged_MEAN_CORPUSCULAR_HEMOGLOBIN_CONC',
 'merged_RED_CELL_DISTRIBUTION_WIDTH',
 'merged_RETICULOCYTE_ABSOLUTE_COUNT',
 'merged_RETICULOCYTE_COUNT_PCT',
 'merged_WHITE_BLOOD_CELLS',
 'merged_PLATELETS',
 'merged_GLUCOSE',
 'merged_CALCIUM',
 'merged_FREE_CALCIUM',
 'merged_SODIUM',
 'merged_POTASSIUM',
 'merged_Bicarbonate',
 'merged_CHLORIDE',
 'merged_BMC_UREA_NITROGEN_BUN',
 'merged_CREATININE',
 'merged_CRP',
 'merged_FERRITIN',
 'merged_LD',
 'merged_FIBRINOGEN',
 'merged_TROPONIN_I',
 'merged_B_TYPE_NATRIURETIC_PEPTIDE',
 'merged_ANION_GAP_WITHOUT_POTASSIUM',
 'merged_ALBUMIN']

prot_labs=[i+'_1tau' for i in prot_labs]
prot_labs=[i for i in prot_labs if i in df_ori.columns]
df_prot[prot_labs]=df_ori[prot_labs]


df_coef_,metrics_df = tr_predict(df_prot, col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=metrics_df.rename(index={0: 'LR-L1-protocol'})# df_AUCs=pd.concat([df_AUCs,df_AUC.rename(index={0: 'LR-L1'})])


# LSTM-transformer score

df_vitals=df_ori[['flag_ICU','vitals_lstm_2020','vitals_lstm_2021','vitals_lstm_2022','vitals_lstm_2023','vitals_lstm_2024','vitals_lstm']]
df_coef_,metrics_df = tr_predict(df_vitals, col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'LR-L1-lstm-score'})])


df_AUCs


# NEWS and qSOFA

df_NEWS=pd.read_csv(DATA_PATH + 'NEWS_score.csv',index_col=0)
df_NEWS=df_NEWS[['flag_ICU','NEWS_score']]

df_coef_,metrics_df = tr_predict1(df_NEWS, col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=metrics_df.rename(index={0: 'LR-L1-NEWS'})


df_qSOFA=pd.read_csv(DATA_PATH + 'qSOFA_score.csv',index_col=0)
df_qSOFA=df_qSOFA[['flag_ICU','qSOFA_score']]

df_coef_,metrics_df = tr_predict1(df_qSOFA, col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring)
df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'LR-L1-qSOFA'})])


df_AUCs



# Parsimonious model

from warnings import filterwarnings
filterwarnings('ignore')

df_new_backup=df_new.copy()


df_census_add=df_census[['PID']+[i for i in ['Total COVID'] if i not in df_new.columns]]
df_new=df_new.merge(df_census_add,how='left', on='PID')


df_new=df_drop(df_new, [])
df_new.head()



my_penalty='l1'

Xraw=df_new.drop([col_y]+['vitals_lstm_2020','vitals_lstm_2021','vitals_lstm_2022','vitals_lstm_2023','vitals_lstm_2024'], axis=1).values#
print(Xraw.shape)

y= df_new[col_y].values
names=df_new.drop([col_y]+['vitals_lstm_2020','vitals_lstm_2021','vitals_lstm_2022','vitals_lstm_2023','vitals_lstm_2024'], axis=1).columns
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(Xraw)



AUC_best=0.5
for my_C in np.linspace(0.01, 0.2, num=20):
#for my_C in [0.1]:
    print(my_C)
    clf=LogisticRegression(C=my_C,penalty=my_penalty,class_weight= 'balanced',solver='liblinear')#0.
    for n_select in range(1,11):
        rfe = RFE(clf, n_select, step=1)
        rfe.fit(X, y.ravel())
        id_keep_1st= list(names[rfe.ranking_==1].values)
        #print(id_keep_1st)
        feature_list=id_keep_1st+[col_y]
        if 'vitals_lstm' in feature_list:
            feature_list+=['vitals_lstm_2020','vitals_lstm_2021','vitals_lstm_2022','vitals_lstm_2023','vitals_lstm_2024']
        df_coef_,metrics_df = tr_predict(df_new[feature_list], col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring, report=True)#test_size=0.2,
        # # metrics_df.describe()
        # # print(n_select) 
        AUC_=metrics_df['AUC']['mean'].values
        if AUC_>AUC_best:
            AUC_best=AUC_
            id_keep_1st_best=id_keep_1st
            print(n_select, AUC_,id_keep_1st_best)




df_coef_,metrics_df = tr_predict(df_new[id_keep_1st_best+[col_y]+['vitals_lstm_2020','vitals_lstm_2021','vitals_lstm_2022','vitals_lstm_2023','vitals_lstm_2024']], col_y=col_y,target_names = ['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring=my_scoring,pred_score=False)#test_size=0.2,, report=True
# metrics_df.describe()
if 'df_AUCs' in globals():
  df_AUCs=pd.concat([df_AUCs,metrics_df.rename(index={0: 'LR-L1-top'})])
else:
  df_AUCs=metrics_df.rename(index={0: 'LR-L1-top'})


df_AUCs


result10=df_coef_.merge(result,how='left', on='Variable')
result10











