lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="reconstruction" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_Reconstruction_SegResNetDS/epoch=24-step=12500-v1.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="reconstruction_test_set" -vars#wandb_tag='$["totalseg","test","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_Reconstruction_SegResNetDS/epoch=24-step=12500-v1.ckpt" --trainer#devices=1

lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="simclr" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimCLR_Cross_Sample_SegResNetDS/epoch=24-step=12500-v1.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="simclr" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimCLR_Cross_Sample_SegResNetDS/epoch=24-step=12500-v1.ckpt" --trainer#devices=1

lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="intra_sample_simclr" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimCLR_SegResNetDS/epoch=24-step=12500.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="intra_sample_simclr" ---vars#wandb_tag='$["totalseg","test","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimCLR_SegResNetDS/epoch=24-step=12500.ckpt" --trainer#devices=1

lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="simsiam" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimSiam_SegResNetDS/epoch=24-step=12500.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="simsiam" --vars#wandb_tag='$["totalseg","test","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimSiam_SegResNetDS/epoch=24-step=12500.ckpt" --trainer#devices=1

lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="simsiam_intrasample" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimSiam_Intrasample_SegResNetDS/epoch=24-step=12500.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="simsiam_intrasample" --vars#wandb_tag='$["totalseg","test","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_SimSiam_Intrasample_SegResNetDS/epoch=24-step=12500.ckpt" --trainer#devices=1

lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="vicreg" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_VicReg_SegResNetDS/epoch=24-step=12500.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="vicreg" --vars#wandb_tag='$["totalseg","test","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_VicReg_SegResNetDS/epoch=24-step=12500.ckpt" --trainer#devices=1

lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="vicreg_intrasample" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_VicReg_Intrasample_SegResNetDS/epoch=24-step=12500.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="vicreg_intrasample" --vars#wandb_tag='$["totalseg","test","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_VicReg_Intrasample_SegResNetDS/epoch=24-step=12500.ckpt" --trainer#devices=1

lighter fit --config=./evaluation/totalseg.yaml --vars#percentage=5 --vars#name="conrecon" --vars#wandb_tag='$["totalseg","fit","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_ConRecon_SegResNetDS/epoch=24-step=12500.ckpt"
lighter test --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_test_overrides.yaml --vars#name="conrecon" --vars#wandb_tag='$"totalseg",["test","decoder-only","5%"]' --vars#project="technical_eval" --system#model#trunk#ckpt_path="/mnt/data1/CT_FM/technical_ablation/runs/checkpoints/CT_FM_ConRecon_SegResNetDS/epoch=24-step=12500.ckpt" --trainer#devices=1