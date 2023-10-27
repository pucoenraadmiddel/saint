import pandas as pd
import numpy as np
import random
random.seed(42)

import argparse
import torch
from torch import nn

from models import SAINT
import torch
import wandb

import sys
sys.argv = ['']
from data_openml import data_split

import torch.optim as optim


print('Reading the data...')
train = pd.read_parquet(r'/home/coenraadmiddel/Documents/RossmannStoreSales/TabNet/tabnet/train_processed.parquet')
print("Read:", train.shape)

#select only a couple of columns

train = train[['Store',
                'DayOfWeek',
                'Promo',
                'StateHoliday',
                'SchoolHoliday',
                'StoreType',
                'Assortment',
                'CompetitionDistance',
                'Promo2SinceWeek',
                'Promo2SinceYear',
                'Year',
                'Month',
                'Day',
                'WeekOfYear',
                'CompetitionOpen',
                'PromoOpen',
                'IsPromoMonth',
                'Sales',
                'Set']]


if "Set" not in train.columns:
    train.reset_index(inplace=True, drop=True)
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index

categorical_columns = ['Store',
                        'DayOfWeek',
                        'Promo',
                        'StateHoliday',
                        'SchoolHoliday',
                        'StoreType',
                        'Assortment',
                        # 'Year',
                        # 'Month',
                        # 'Day',
                        # 'WeekOfYear',
                        'IsPromoMonth']

# split x and y
X_all, y_all = train.drop(columns = ['Sales', 'Set']), np.log1p(train[['Sales']].values)

temp = X_all.fillna("MissingValue")
nan_mask = temp.ne("MissingValue").astype(int)

X_train_d, y_train_d = data_split(X_all, y_all, nan_mask, train_indices)
X_valid_d, y_valid_d = data_split(X_all, y_all, nan_mask, valid_indices)
X_test_d, y_test_d = data_split(X_all, y_all, nan_mask, test_indices)


X_train = X_train_d['data']
X_test = X_test_d['data']
X_valid = X_valid_d['data']

y_train = y_train_d['data']
y_test = y_test_d['data']
y_valid = y_valid_d['data']

#force categorical columns to the categorical type

train[categorical_columns] = train[categorical_columns].astype('category')

#get the indices of the categorical columns in train dataFrame

cat_idxs = [train.columns.get_loc(c) for c in categorical_columns if c in train]

#get the dimensions of the categorical columns in train dataFrame

cat_dims = [len(train[c].cat.categories) for c in categorical_columns if c in train]

y_dim = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {}".format(device))


cont_idxs = [i for i in range(X_train.shape[1]) if i not in cat_idxs]
train_mean, train_std = np.array(X_train_d['data'][:,cont_idxs],dtype=np.float32).mean(0), np.array(X_train_d['data'][:,cont_idxs],dtype=np.float32).std(0)
cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

print('finished loading...')

def main(opt):
    wandb.init(project="saint_rossmann_mse", group='seed=7', config=opt)
    #for regression this is the output dimension
    
    opt = wandb.config

    #set the initialization seed:
    torch.manual_seed(7)

    print(opt)
    model = SAINT(categories = tuple(cat_dims), 
                    num_continuous = len(cont_idxs),                
                    dim = opt.embedding_size,                           
                    dim_out = 1,                       
                    depth = opt.transformer_depth,                       
                    heads = opt.attention_heads,                         
                    attn_dropout = opt.attention_dropout,             
                    ff_dropout = opt.ff_dropout,                  
                    mlp_hidden_mults = (4, 2),       
                    cont_embeddings = opt.cont_embeddings,
                    attentiontype = opt.attentiontype,
                    final_mlp_style = opt.final_mlp_style,
                    y_dim = y_dim,
                    )


    criterion = nn.MSELoss().to(device)
    # criterion = nn.PoissonNLLLoss().to(device)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
    best_valid_auroc = 0
    best_valid_accuracy = 0
    best_test_auroc = 0
    best_test_accuracy = 0
    best_valid_rmse = 100000
    print('Using the optimizer: ', opt.optimizer)
    print('Training begins now.')

    from torch.utils.data import DataLoader

    from data_openml import DataSetCatCon

    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32) 

    train_ds = DataSetCatCon(X_train_d, y_train_d, cat_idxs, task='regression', continuous_mean_std=continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True)

    valid_ds = DataSetCatCon(X_valid_d, y_valid_d, cat_idxs, task='regression', continuous_mean_std=continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False)

    test_ds = DataSetCatCon(X_test_d, y_test_d, cat_idxs, task='regression', continuous_mean_std=continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False)

    # if opt.pretrain:
    #     from pretraining import SAINT_pretrain
    #     model = SAINT_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)

    from augmentations import embed_data_mask
    from utils import count_parameters, classification_scores, mean_sq_error
    import os

    modelsave_path = os.path.join(opt.savemodelroot, opt.run_name)

    vision_dset = opt.vision_dset

    for epoch in range(opt.epochs):
        print('Starting epoch: ', epoch)
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            
            y_outs = model.mlpfory(y_reps)
            if opt.task == 'regression':
                loss = criterion(y_outs, y_gts) 
            else:
                loss = criterion(y_outs,y_gts.squeeze()) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(running_loss)
        if opt.active_log:
            wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
            'loss': loss.item()
            })
        if epoch%1==0:
                model.eval()
                with torch.no_grad():
                    valid_rmse, orig_valid_rmse = mean_sq_error(model, validloader, device, vision_dset)    
                    test_rmse, orig_test_rmse = mean_sq_error(model, testloader, device, vision_dset)  
                    train_rmse, orig_train_rmse = mean_sq_error(model, trainloader, device, vision_dset)  
                    print('[EPOCH %d] VALID RMSE: %.3f, ORIG VALID RMSE: %.3f' %
                        (epoch + 1, valid_rmse, orig_valid_rmse ))
                    print('[EPOCH %d] TEST RMSE: %.3f, ORIG TEST RMSE: %.3f' %
                        (epoch + 1, test_rmse, orig_test_rmse ))
                    print('[EPOCH %d] TRAIN RMSE: %.3f, ORIG TRAIN RMSE: %.3f' %
                        (epoch + 1, train_rmse, orig_train_rmse ))
                    
                    if opt.active_log:
                        wandb.log({'valid_rmse': valid_rmse
                                    , 'test_rmse': test_rmse
                                    , 'train_rmse': train_rmse
                                    , 'orig_valid_rmse': orig_valid_rmse
                                    , 'orig_test_rmse': orig_test_rmse
                                    , 'orig_train_rmse': orig_train_rmse })     
                    if valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        best_test_rmse = test_rmse
                        best_train_rmse = train_rmse
                        
                        #get the run id from wandb
                        run_id = wandb.run.id
                        torch.save(model.state_dict(), f'{modelsave_path}/SAINT_model_best_{run_id}.pt')
                        #Save as artifact on weights and biases
                        artifact = wandb.Artifact(f'SAINT_model_best_{run_id}', type='model')
                        #add the run id to the artifact                       
                        wandb.run.log_artifact(artifact)
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                        print(f"Early stopping counter: {early_stop_count}/{opt.patience}")
                        if early_stop_count >= opt.patience:
                            print('EARLY STOPPING')
                            break
                model.train()
                
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    if opt.task =='binary':
        print('AUROC on best model:  %.3f' %(best_test_auroc))
    elif opt.task =='multiclass':
        print('Accuracy on best model:  %.3f' %(best_test_accuracy))
    else:
        print('RMSE on best model:  %.3f' %(best_test_rmse))

    if opt.active_log:
        if opt.task == 'regression':
            wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep':best_test_rmse, 
            'cat_dims':len(cat_idxs) , 'con_dims':len(cont_idxs), 'valid_rmse_best': best_valid_rmse })        
        else:
            wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc , 
            'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(cont_idxs) })

            
            
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    # parser.add_argument('--dset_id', required=True, type=int)
    parser.add_argument('--vision_dset', action = 'store_true')
    parser.add_argument('--task', default='regression', type=str,choices = ['binary','multiclass','regression'])
    parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=3, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

    parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--savemodelroot', default='/home/coenraadmiddel/Documents/RossmannStoreSales/SAINT/saint/bestmodels/regression', type=str)
    parser.add_argument('--run_name', default='rossmann_local', type=str)
    parser.add_argument('--set_seed', default= 3 , type=int)
    parser.add_argument('--dset_seed', default= 42 , type=int)
    parser.add_argument('--active_log', default=True, type=bool)

    parser.add_argument('--pretrain', action = 'store_true')
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
    parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    # parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)

    parser.add_argument('--ssl_avail_y', default= 0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

    parser.add_argument('--patience', default=10, type=int, help='Number of epochs to wait before stopping if no improvement in validation accuracy')


    opt = parser.parse_args()   

    main(opt)