#!/bin/bash

python segment_process.py --data_dir "./a_dataset/english/" --window_length 1.0 --model_type "tiny" --overlap 0.0 --run_device "cpu"
# Explain of args in segment_process.py file
# options:
#   -h, --help            show this help message and exit
#   --data_dir DATA_DIR   dataset directory
#   --model_type MODEL_TYPE Whisper model verion 
#   --run_device RUN_DEVICE run device
#   --window_length WINDOW_LENGTH window length
#   --overlap OVERLAP     overlap

python train_MixSAE.py  --data_dir "./a_dataset/english/" --window_length 1.0 --collar 0.0 --model_type "tiny" --pretrain_epochs_main 30 --pretrain_epochs 20 --main_train_epochs 20
# Explain of args in train_MOE_CL.py file
# MixSAE Network Parameters

# options:
#   -h, --help            show this help message and exit
#   --data_dir DATA_DIR   dataset directory
#   --n-classes N_CLASSES
#                         output dimension - n_speakers (default: 2)
#   --lr LR               learning rate (default: 1e-4)
#   --wd WD               weight decay (default: 5e-4)
#   --batch-size BATCH_SIZE
#                         input batch size for training
#   --model_type MODEL_TYPE
#                         model type (default: "tiny")
#   --hidden-dims HIDDEN_DIMS
#                         hidden dimensions for autoencoders (default: [256, 128, 64, 32])
#   --latent_dim LATENT_DIM
#                         latent space dimension
#   --n-clusters N_CLUSTERS
#                         number of clusters in the latent space
#   --n-jobs N_JOBS       number of jobs to run in parallel
#   --log-interval LOG_INTERVAL
#                         how many batches to wait before logging the training status
#   --window_length WINDOW_LENGTH
#                         window length
#   --overlap OVERLAP     overlap
#   --rho RHO             Sparsity hyperparameter of single sparse autoencoder
#   --pretrain_epochs PRETRAIN_EPOCHS
#                         epochs for pretraining k-autoencoders
#   --pretrain_epochs_main PRETRAIN_EPOCHS_MAIN
#                         epochs for pretraining the main autoencoder for the whole dataset
#   --pretrain PRETRAIN   whether use pre-training
#   --main_train_epochs MAIN_TRAIN_EPOCHS
#                         epochs for main-training phase
#   --sparsity_param SPARSITY_PARAM
#                         sparsity lost param
#   --cl_loss_param CL_LOSS_PARAM
#                         clasification loss param
#   --collar COLLAR       collar param for DER