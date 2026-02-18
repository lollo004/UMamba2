cd /workspace/pulpy
source /workspace/pulpy/.venv/bin/activate
source /workspace/pulpy/code/UMamba2/nnUNet_env.sh

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 5 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans_torchres -tr nnUNetTrainer_Pulpy_accum2
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 5 3d_fullres_torchres_mambabot_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans_torchres -tr nnUNetTrainer_Pulpy_accum2