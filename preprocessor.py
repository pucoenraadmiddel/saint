

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_openml import data_split
from torch.utils.data import DataLoader
from data_openml import DataSetCatCon

import pandas as pd
import numpy as np

#Define a preprocessor class taking in the data and the target column and returning the model specific preprocessed data

class Preprocessor:
    '''This class takes in a df_train as training data, df_val as the validation data and df_test as the test data.
    It performs model specific (CatBoost, XGBoost, TabNet and SAINT Regressor models) preprocessing
    and returns the preprocessed data as a dictionary.
    
    The random state is fixed at 42.
    '''
    
    def __init__(self, df, target_col='Frequency'):
        self.df = df 
        self.cat_cols = ['CHA_SubChannel',
                            'Cover_Type',
                            'DRI_BadGroup',
                            'DRI_LicenseCode',
                            'DRI_Gender', 
                            'DRI_MaritalStatus',

                            'LOC_AccRateId',
                            'LOC_AvgITCSuburb_V2',
                            'LOC_CrestaZone_V',
                            'LOC_IncomeSuburb2018',
                            'LOC_Languages_V2',
                            'LOC_NightParkingLocation',
                            'LOC_PopulationDensity_V2',
                            'LOC_ProvinceNight',
                            'LOC_SettlementSuburb_V2',
                            'LOC_TheftFromMotorLocation2018',

                            'POL_AccClaimRecency_CAT',
                            'POL_Blacklist23',    

                            'VEH_BodyType',
                            'VEH_Colour',
                            'VEH_Country',
                            'VEH_ManualAuto',
                            'VEH_ModelGroupId',
                            'VEH_Usage',
                            
                            'POL_ITCLapse_CAT',
                            'POL_ITCScore_CAT',
                            
                            'LOC_DayParkingLocation',
                            'LOC_Genders_V2',
                            'LOC_HouseDensity_V2',
                            
                            'POL_TheftClaimRecency_CAT',
                            'DRI_CompInsuranceDuration',

                                ]
        
        self.ord_cols = ['POL_AssetFinance',
                            'POL_CarsOnCover',
                            'POL_ClaimCount6Glass',
                            'POL_ClaimCountAccAll',
                            'POL_FinanceTotal',
                            'POL_Insurance',
                            'POL_AccClaimRecency_ORD',

                            'VEH_TravelDistance',

                            'POL_ITCScore_ORD',
                            'POL_ITCLapse_ORD',
                            
                            'DRI_Age',
                            'DRI_AgeAtIssue',
                            'DRI_LicenseYears_V',
                            
                            'POL_TheftClaimRecency_ORD',
                            
                            'VEH_CarAge',
                            'VEH_Excess',
                            'VEH_Kilowatts',
                            'VEH_SumInsured',
                            'VEH_NoGears',
                            'VEH_Mileage',
                            
                            ]
        
        self.training_cols = self.cat_cols + self.ord_cols
               
        self.df = self.imputer(self.df)   
        
        self.df[self.cat_cols] = self.df[self.cat_cols].astype('category')
        self.df[self.ord_cols] = self.df[self.ord_cols].astype('float32')   
        
        # self.df = self.df.sort_values(by = 'TIM_Month') 
        
        #This is my own out of time sample. I want to use the one from Mjolnir 4 based on their random number
        # self.df_test = self.df.tail(int(len(self.df)*0.1))
        
        # self.df_train, self.df_val = train_test_split(self.df.head(int(len(self.df)*0.9)), test_size=(self.df_test.shape[0]/(self.df.shape[0] - self.df_test.shape[0])), random_state=42)
        
        ix_train = self.df['RandomNumber'].isin([1,2,3,4,5,6,7,8])
        ix_val = self.df['RandomNumber'].isin([9])
        ix_test = self.df['RandomNumber'].isin([10])

        #Create dfs with all variables, including those not modelled on 
        self.df_train = self.df[ix_train]
        self.df_val = self.df[ix_val]
        self.df_test = self.df[ix_test]
        
        self.target_col = target_col 
        self.df_target = self.df[self.target_col]
        
        #create X and y train with only the variables modelled on
        self.X_train = self.df_train[self.training_cols]
        self.y_train = self.df_train[self.target_col]
        
        #create X and y train with only the variables modelled on
        self.X_val = self.df_val[self.training_cols]
        self.y_val = self.df_val[self.target_col]

        #create X and y train with only the variables modelled on
        self.X_test = self.df_test[self.training_cols]
        self.y_test = self.df_test[self.target_col]
        
        self.cat_idxs = [self.X_train.columns.get_loc(c) for c in self.cat_cols if c in self.X_train]
        
        self.cat_dims = [len(self.X_train[c].cat.categories) for c in self.cat_cols if c in self.X_train]
        
        self.ord_idxs = [self.X_train.columns.get_loc(c) for c in self.ord_cols if c in self.X_train]   
        
    def imputer(self, df, impute_dict = {'POL_Blacklist23': 'None'
                                         , 'Acc_GrossClaim': 0}, verbose = True):
        '''imputes the missing values in the categorical columns with the mode of the column or a specified value from the impute_dict dictionary.
        Also imputes the missing values in the numerical columns with the mean of the column or a specified value from the impute_dict dictionary.'''
        for col in self.cat_cols:
            if col in impute_dict.keys():
                df[col] = df[col].fillna(impute_dict[col])
                if verbose:
                    print(f'Imputed {col} with {impute_dict[col]}')
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                if verbose:
                    print(f'Imputed {col} with the mode: {df[col].mode()[0]}')
                    
        for col in self.ord_cols:
            if col in impute_dict.keys():
                df[col] = df[col].fillna(impute_dict[col])
                if verbose:
                    print(f'Imputed {col} with {impute_dict[col]}')
            else:
                df[col] = df[col].fillna(df[col].mean())
                if verbose:
                    print(f'Imputed {col} with the mean: {df[col].mean()}')
                 
        return df
        
    def preprocess_for_catboost(self):
        '''Preprocesses the data for CatBoostRegressor model.
        Returns the preprocessed data as a dictionary.
        This can be directly used in the .fit method.
        '''
        
        #label encoding for catboost
        
        self.X_train_catboost = self.X_train.copy(deep=True)
        self.y_train_catboost = self.y_train.copy(deep=True)                                                                                                      
        
        self.X_val_catboost = self.X_val.copy(deep=True)
        self.y_val_catboost = self.y_val.copy(deep=True)
                                                    
        for col in self.cat_cols:
            print(col, self.X_train_catboost[col].nunique())
            self.X_train_catboost[col] = self.X_train_catboost[col].astype(str)
            self.X_val_catboost[col] = self.X_val_catboost[col].astype(str)
            l_enc = LabelEncoder()
            l_enc.fit(self.X_train_catboost[col].values)
            
            self.X_train_catboost[col] = l_enc.transform(self.X_train_catboost[col].values)
            self.X_val_catboost[col] = l_enc.transform(self.X_val_catboost[col].values)
                    
        fit_dict = {'X': self.X_train_catboost,
                    'y': self.y_train_catboost,
                    'cat_features': self.cat_idxs,
                    'eval_set' : (self.X_val_catboost, self.y_val_catboost),
                    }

        return fit_dict

    def preprocess_for_xgboost(self,  train_size = None):
        '''Preprocesses the data for XGBoostRegressor model.
        Returns the preprocessed data as a dictionary.
        '''
        if train_size is None:
            train_size = int(self.X_train.shape[0]/10) 
            print('Using only 10% of the training data for XGBoost')
            
        print('Training data size:', train_size)
        
        import pandas as pd
        import xgboost as xgb
        
        self.X_train_xgboost = self.X_train[: train_size].copy(deep=True)
        self.y_train_xgboost = self.y_train[: train_size].copy(deep=True)                                                                                                     
        
        self.X_val_xgboost = self.X_val.copy(deep=True)
        self.y_val_xgboost = self.y_val.copy(deep=True)

        self.X_test_xgboost = self.X_test.copy(deep=True)
        self.y_test_xgboost = self.y_test.copy(deep=True)
        
        for col in self.cat_cols:
            print(col, self.X_train_xgboost[col].nunique())
            self.X_train_xgboost[col] = self.X_train_xgboost[col].astype(str)
            self.X_val_xgboost[col] = self.X_val_xgboost[col].astype(str)
            self.X_test_xgboost[col] = self.X_test_xgboost[col].astype(str)
            self.df[col] = self.df[col].astype(str)
            l_enc = LabelEncoder()
            l_enc.fit(self.df[col].values)
            
            self.X_train_xgboost[col] = l_enc.transform(self.X_train_xgboost[col].values)
            self.X_val_xgboost[col] = l_enc.transform(self.X_val_xgboost[col].values)
            self.X_test_xgboost[col] = l_enc.transform(self.X_test_xgboost[col].values)
            
        
        dtrain = xgb.DMatrix(self.X_train_xgboost, self.y_train_xgboost, enable_categorical=True)
        dval = xgb.DMatrix(self.X_val_xgboost, self.y_val_xgboost, enable_categorical=True)
        dtest = xgb.DMatrix(self.X_test_xgboost, self.y_test_xgboost, enable_categorical=True)
                            
        preprocessed_data = {'dtrain': dtrain, 'dval': dval, 'dtest': dtest}
        return preprocessed_data
    
    def preprocess_for_tabnet(self, train_size = None):
        '''Preprocesses the data for TabNetRegressor model.
        Returns the preprocessed data as a dictionary.
        '''
        
        if train_size is None:
            train_size = int(self.X_train.shape[0]/10) 
            print('Using only 10% of the training data for TabNet')
            
        print('Training data size:', train_size)
        self.X_train_tabnet = self.X_train[: train_size].copy(deep=True)
        self.y_train_tabnet = self.y_train[: train_size].copy(deep=True)                                                                                                     
        
        self.X_val_tabnet = self.X_val.copy(deep=True)
        self.y_val_tabnet = self.y_val.copy(deep=True)

        self.X_test_tabnet = self.X_test.copy(deep=True)
        self.y_test_tabnet = self.y_test.copy(deep=True)
        
        for col in self.cat_cols:
            print(col, self.X_train_tabnet[col].nunique())
            self.X_train_tabnet[col] = self.X_train_tabnet[col].astype(str)
            self.X_val_tabnet[col] = self.X_val_tabnet[col].astype(str)
            self.X_test_tabnet[col] = self.X_test_tabnet[col].astype(str)
            self.df[col] = self.df[col].astype(str)
            l_enc = LabelEncoder()
            l_enc.fit(self.df[col].values)
            
            self.X_train_tabnet[col] = l_enc.transform(self.X_train_tabnet[col].values)
            self.X_val_tabnet[col] = l_enc.transform(self.X_val_tabnet[col].values)
            self.X_test_tabnet[col] = l_enc.transform(self.X_test_tabnet[col].values)
        
        
        preprocessed_data = {'X_train' : self.X_train_tabnet.values, 
                             'X_val' : self.X_val_tabnet.values,
                             'X_test' : self.X_test_tabnet.values,
                             'y_train' :self.y_train_tabnet.values.reshape(-1, 1), 
                             'y_val': self.y_val_tabnet.values.reshape(-1, 1),
                             'y_test': self.y_test_tabnet.values.reshape(-1, 1),
                             'X_all' : np.concatenate([self.X_train_tabnet.values, self.X_val_tabnet.values, self.X_test_tabnet.values]),
                             'y_all' : np.concatenate([self.y_train_tabnet.values.reshape(-1, 1), self.y_val_tabnet.values.reshape(-1, 1), self.y_test_tabnet.values.reshape(-1, 1)])
        }
        return preprocessed_data 

    def preprocess_for_saint(self, batch_size = 64, load_percentage = 0.1):
        '''Preprocesses the data for SAINTRegressor model.'''    
        
        #get all indexes for the self.training_cols cols
        
        X_all = self.df[self.training_cols].copy()
        y_all = self.df[self.target_col].copy().values
        
        #label encoding for categorical features
        for col in self.cat_cols:
            print(col, X_all[col].nunique())
            X_all[col] = X_all[col].astype(str)
            self.df[col] = self.df[col].astype(str)
            l_enc = LabelEncoder()
            l_enc.fit(self.df[col].values)
            
            X_all[col] = l_enc.transform(X_all[col].values)
        
        # temp = X_all.fillna("MissingValue")
        
        #Assert if there are any missing values in X_all
        assert X_all.isnull().sum().sum() == 0, "There are missing values in X_all"
        
        temp = X_all
        nan_mask = X_all.ne("MissingValue").astype(int)
                    
        ix_train = self.df['RandomNumber'].isin([1,2,3,4,5,6,7,8]).index
        ix_val = self.df['RandomNumber'].isin([9]).index
        ix_test = self.df['RandomNumber'].isin([10]).index  
        
        print(f'loading only the first {load_percentage*100}% of the training data')
        ix_train = ix_train[:int(len(ix_train)*load_percentage)]
        X_train_d, y_train_d = data_split(X_all, y_all, nan_mask, ix_train)
        
        X_valid_d, y_valid_d = data_split(X_all, y_all, nan_mask, ix_val)
        X_test_d, y_test_d = data_split(X_all, y_all, nan_mask, ix_test)

        
        train_mean, train_std = np.array(X_train_d['data'][:, self.ord_idxs],dtype=np.float32).mean(0), np.array(X_train_d['data'][:, self.ord_idxs],dtype=np.float32).std(0)
        continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32) 
        
        train_ds = DataSetCatCon(X_train_d, y_train_d, self.cat_idxs, task='regression', continuous_mean_std=continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        valid_ds = DataSetCatCon(X_valid_d, y_valid_d, self.cat_idxs, task='regression', continuous_mean_std=continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

        test_ds = DataSetCatCon(X_test_d, y_test_d, self.cat_idxs, task='regression', continuous_mean_std=continuous_mean_std)
        testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        
        return trainloader, validloader, testloader
