## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt"

# ######################### Decoder-only ############################
lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/suprem_unet.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="suprem_unet" --vars#project="word" --vars#wandb_group='representations'; pkill -9 -f lighter
lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresencoder.yaml,./evaluation/overrides/representation_head.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="ct_fm" --vars#project="word" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='representations'
lighter fit --config=./evaluation/word.yaml,./evaluation/baselines/segresnet_vista3d.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="vista3d" --vars#project="word" --vars#wandb_group='representations'; pkill -9 -f lighter