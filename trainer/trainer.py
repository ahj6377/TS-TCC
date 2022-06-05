import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss



def Trainer(model, criterion, scheduler, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")


    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        """
        if 1<= epoch and epoch <= 5:
            tmplr = -(5e-5) * (epoch - 11)
            model_optimizer = torch.optim.Adam(model.parameters(), lr=tmplr, betas=(0.9,0.99), weight_decay=3e-4)
            temp_cont_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=tmplr, betas=(0.9,0.99), weight_decay=3e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
        """
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    accval=[]
    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')
        accval = [test_loss, test_acc]

    logger.debug("\n################## Training is Done! #########################")
    return accval


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    #for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
    for batch_idx, (data, labels, aug1, aug3, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        #aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        aug1, aug3, aug2 = aug1.float().to(device), aug3.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
            predictions2, features3 = model(aug3)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)
            features3 = F.normalize(features3, dim=1)

            #temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            #temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            temp_cont_loss12, temp_cont_lstm_feat12, tloss12, f12 = temporal_contr_model(features1, features2)
            temp_cont_loss21, temp_cont_lstm_feat21, tloss21, f21 = temporal_contr_model(features2, features1)

            temp_cont_loss13, temp_cont_lstm_feat13, tloss13, f13 = temporal_contr_model(features1, features3)
            temp_cont_loss31, temp_cont_lstm_feat31, tloss31, f31 = temporal_contr_model(features3, features1)

            temp_cont_loss23, temp_cont_lstm_feat23, tloss23, f23 = temporal_contr_model(features2, features3)
            temp_cont_loss32, temp_cont_lstm_feat32, tloss32, f32 = temporal_contr_model(features3, features2)

            # normalize projection feature vectors

            #zis = temp_cont_lstm_feat1
            #zjs = temp_cont_lstm_feat2

            zis12 = temp_cont_lstm_feat12 
            zjs21 = temp_cont_lstm_feat21 

            zis13 = temp_cont_lstm_feat13 
            zjs31 = temp_cont_lstm_feat31 

            zis23 = temp_cont_lstm_feat23 
            zjs32 = temp_cont_lstm_feat32 

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            lambda11 = 0.5
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)

            #loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2

            #loss1 = (temp_cont_loss12 + temp_cont_loss21) * lambda1 +  nt_xent_criterion(zis12, zjs21) * lambda2
            loss1 = (temp_cont_loss12 + temp_cont_loss21) * lambda1 + (tloss12 + tloss21) * lambda11 + ( 0.5 * nt_xent_criterion(zis12, zjs21) + 0.5 * nt_xent_criterion(f12,f21)) * lambda2

            #loss2 = (temp_cont_loss13 + temp_cont_loss31) * lambda1 +  nt_xent_criterion(zis13, zjs31) * lambda2
            loss2 = (temp_cont_loss13 + temp_cont_loss31) * lambda1 + (tloss13 + tloss31) * lambda11 + ( 0.5 * nt_xent_criterion(zis13, zjs31) + 0.5 * nt_xent_criterion(f13,f31)) * lambda2

            #loss3 = (temp_cont_loss23 + temp_cont_loss32) * lambda1 +  nt_xent_criterion(zis23, zjs32) * lambda2
            loss3 = (temp_cont_loss23 + temp_cont_loss32) * lambda1 + (tloss23 + tloss32) * lambda11 + ( 0.5 * nt_xent_criterion(zis23, zjs32) + 0.5 * nt_xent_criterion(f23,f32)) * lambda2
            loss = max(loss1, loss2, loss3)
            
        else: # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        #for data, labels, _, _ in test_dl:
        for data, labels, _, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs



