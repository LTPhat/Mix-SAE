import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from pyannote.metrics.diarization import DiarizationErrorRate
import matplotlib.pyplot as plt
import os
import pandas as pd
from mixture_of_autoencoders import MoESparseAutoencodersCL
from load_dataset import CustomDataset
from create_DER import createDER


######### NOTE: ARGUMENT ########################
parser = argparse.ArgumentParser(description='MixSAE Network')

# Dataset parameters
parser.add_argument('--data_dir', default='./a_dataset/english/',
                    help='dataset directory')
parser.add_argument('--input_dim', type=int, default=384,
                    help='input dimension (based on Whisper version output)')
parser.add_argument('--n-classes', type=int, default=2,
                    help='output dimension')

# Training parameters
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')

# Model parameters
parser.add_argument('--hidden-dims', default=[256, 128, 64, 32],
                    help='learning rate (default: 1e-4)')
parser.add_argument('--latent_dim', type=int, default=2,
                    help='latent space dimension')
parser.add_argument('--n-clusters', type=int, default=2,
                    help='number of clusters in the latent space')


# Utility parameters
parser.add_argument('--n-jobs', type=int, default=1,
                    help='number of jobs to run in parallel')
parser.add_argument('--log-interval', type=int, default=20,
                    help=('how many batches to wait before logging the '
                            'training status'))
parser.add_argument("--window_length", type = float, default= 0.2, help="window length")
parser.add_argument("--overlap", type = float, default= 0, help="overlap")
parser.add_argument('--rho', type=float, default=0.2,
                    help='Sparsity hyperparameter of single sparse autoencoder')
parser.add_argument('--pretrain_epochs', type=int, default=30,
                    help='epochs for pretraining k-autoencoders')
parser.add_argument('--pretrain_epochs_main', type=int, default= 20,
                    help='epochs for pretraining the main autoencoder for the whole dataset')
parser.add_argument('--pretrain', type=bool, default=True,
                    help='whether use pre-training')
parser.add_argument('--main_train_epochs', type=int, default = 20,
                    help='epochs for main-training phase')
parser.add_argument('--sparsity_param', type=float, default=0.2,
                    help='sparsity lost param')
parser.add_argument('--cl_loss_param', type=float, default= 0.1,
                    help='clasification loss param')
parser.add_argument('--collar', type=float, default= 0.0,
                    help='collar param for DER')

args = parser.parse_args()
label_dir = args.data_dir + "/label"
segment_dir = args.data_dir + "/segments_{}".format(args.window_length)
label_dir = args.data_dir + "/label"
window_length = args.window_length
overlap = args.overlap

sample_list = sorted(os.listdir(segment_dir))
label_list = sorted(os.listdir(label_dir))

def main():
    # Dataframe to save result
    df = pd.DataFrame(columns=["Language", "Filename", "MOE_CL"])

    for sample, label in zip(sample_list, label_list):
        print("Processing segments in folder {}".format(sample))
        print("Label: ", label)
        
        sample_path = segment_dir + "/" + sample
        label_path = label_dir + '/' + label

        DER = DiarizationErrorRate(collar=args.collar, skip_overlap=False)

        # # --------------------------- DEEP CLUSTERING ------------------------------------

        train_dataset = CustomDataset(sample_dir=sample_path, embed_dim=args.input_dim)
        # train_dataset = CustomDataset(sample_dir=sample_path)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        test_dataset = CustomDataset(sample_dir=sample_path, embed_dim=args.input_dim)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= False)

        #------------------MOE SPARITY CL------------------------------------
        moe_cl = MoESparseAutoencodersCL(args=args)
        mixture_moe_cl, full_latent_X = moe_cl.pretraining(train_loader)
        pre_label = moe_cl.psedo_label 
        mixture_moe_cl = moe_cl.main_training(train_loader)
        moe_cl_pred = moe_cl.get_final_cluster(train_loader)

        moe_cl_ref, moe_cl_hyp = createDER(label_path=label_path, sample_dir=sample_path, prediction=pre_label, window_length=window_length, overlap=overlap)
        moe_cl_pre_error = DER(moe_cl_ref, moe_cl_hyp)

        # print("MOE CL PRE DER", moe_cl_pre_error)

        moe_cl_ref, moe_cl_hyp = createDER(label_path=label_path, sample_dir=sample_path, prediction=moe_cl_pred, window_length=window_length, overlap=overlap)
        moe_cl_error = DER(moe_cl_ref, moe_cl_hyp)
        print(f"Sample: {sample}; DER: {moe_cl_error}")

        # Update result to csv
        language = args.data_dir.split("/")[-2]
        file_name = sample
        result_dir = f"./results/{language}/{window_length}/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)


        # Add a new row
        new_row = {"Language": language, "Filename": file_name, "MOE_CL": moe_cl_error}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)        
    
    #-- Save result
    csv_path = f"{result_dir}MixSAE_{language}_{window_length}.csv"
    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_path, index=False)
    # Calculate the average MOE_CL score
    avg_score = df['MOE_CL'].mean()
    json_path = os.path.join(result_dir, "avg_DER.json")
    # Write the average score to the JSON file
    with open(json_path, 'w') as file:
        json.dump({"avg_DER": avg_score}, file, indent=4)
    print("----------Training DONE----------------")

if __name__ == "__main__":
    print(args)
    main()
