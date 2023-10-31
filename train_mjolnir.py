import pandas as pd
import numpy as np
import random
random.seed(42)

import argparse
import torch
from torch import nn

from models import SAINT
from augmentations import embed_data_mask
import torch
import wandb

import torch.optim as optim
from utils import get_loss
import preprocessor as pp



def main(args):
        
    if args.reload_data:
        print('reload_data: ', args.reload_data)
        df = pd.read_parquet('df_post.parquet')
        pre = pp.Preprocessor(target_col = 'Acc_GrossClaim')
        pre.basics(df)

    else:
        print('reload_data: ', args.reload_data)
        pre = pp.Preprocessor(target_col = 'Acc_GrossClaim')
        #load from save_dict 
        pre.cat_idxs, pre.cat_dims, pre.ord_idxs = pre.load_from_save_dict()

        
    cat_idxs = pre.cat_idxs
    cat_dims = pre.cat_dims
    cont_idxs = pre.ord_idxs

    cat_dims_saint = np.append(np.array([1]), np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {}".format(device))

    print("----------------------------------------")
    print("----------------------------------------")
    print('Finished loading... ')
    print("----------------------------------------")
    print("----------------------------------------")
        
    
    wandb.init(project="SAINT_mjolnir", config=args, entity="middelman", group='MSE')
    
    args = wandb.config    

    print(args)
    model = SAINT(categories = tuple(cat_dims_saint), 
                    num_continuous = len(cont_idxs),                
                    dim = args.embedding_size,                           
                    dim_out = 1,                       
                    depth = args.transformer_depth,                       
                    heads = args.attention_heads,                         
                    attn_dropout = args.attention_dropout,             
                    ff_dropout = args.ff_dropout,                  
                    mlp_hidden_mults = (4, 2),    
                    # continuous_mean_std = None, #pre.cont_mean_std,   
                    cont_embeddings = args.cont_embeddings,
                    attentiontype = args.attentiontype,
                    final_mlp_style = args.final_mlp_style,
                    y_dim = 1,
                    )  


    model.to(device)
    optimizer = optim.AdamW(model.parameters()
                            , lr=args.lr
                            , weight_decay=5e-5
                            )
    
    # criterion = nn.MSELoss().to(device)
    criterion = nn.PoissonNLLLoss(log_input=False).to(device)
            
    best_valid_loss = 1000000 #Unrealistically high number
    best_test_loss = 1000000 #Unrealistically high number
    early_stop_count = 0
    print('Using the optimizer: ', args.optimizer)
    print('Training begins now.')         
    print('Reloading data: ', args.reload_data)

    if args.reload_data:
        trainloader, validloader, testloader = pre.preprocess_for_saint(batch_size = args['batchsize'], save = True, load_percentage = 0.2)#, load_percentage = i/5)
    else:
        trainloader, validloader, testloader = pre.load_for_saint()
    
    for epoch in range(args.epochs):
            print('Starting epoch: ', epoch)
            model.train()
            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

                # We are converting the data to embeddings in the next step
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)           
                reps = model.transformer(x_categ_enc, x_cont_enc)
                # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:,0,:]
                
                y_outs = model.mlpfory(y_reps)
                loss = criterion(y_outs, y_gts) 
                loss.backward()
                optimizer.step()
            if args.active_log:
                wandb.log({'epoch': epoch , 
                'train_loss': loss.item()})
            
            print('Finished epoch: ', epoch)
            print('Training loss: ', loss.item())
            
            valid_loss = get_loss(model, validloader, device, criterion)
            test_loss = get_loss(model, testloader, device, criterion)
            
            if args.active_log:
                wandb.log({'valid_loss': valid_loss.item(), 'test_loss': test_loss.item()})

            
            if best_valid_loss > valid_loss:
                print("Validation loss did not improve.")
                early_stop_count += 1
                print(f"Early stopping counter: {early_stop_count}/{args.patience}")
                if early_stop_count >= args['patience']:
                    print('EARLY STOPPING')
                    break
            else:
                print("Validation loss improved!")
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'models/mjolnir4/SAINT_model_best_{run_id}.pt')
                run_id = wandb.run.id
                early_stop_count = 0

            if best_test_loss > test_loss:
                print("testation loss did not improve.")
            else:
                print("testation loss improved!")
                best_test_loss = test_loss
            model.train()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--reload_data', action='store_true')
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

    parser.add_argument('--patience', default=5, type=int, help='Number of epochs to wait before stopping if no improvement in validation accuracy')


    args = parser.parse_args()   

    main(args)
