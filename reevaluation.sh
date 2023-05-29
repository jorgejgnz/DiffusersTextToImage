export DATA_DIR="../experimentos2/data/dataset_4atts"
export ATTESTIMATOR_CHECKPOINT="/home/jorge/UPV/ARF/experimentos2/face_attribute_estimator_output/checkpoint/checkpoint.pth.tar"


export OUTPUT_DIR="re-evaluations/CondDDPM"
export RESULTS_DIR="../RUNPOD/generic_finetunings_4atts/DDPM_batch256*1_lr-cosine-1e-04_epochs10_snrgamma5.0/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}"
  

'''
export OUTPUT_DIR="re-evaluations/UNet_big_RunPod"
export RESULTS_DIR="../RUNPOD/generic_finetunings_4atts/MyDiffusion_batch8*1_lr-cosine-1e-04_epochs10_snrgamma5.0_channels256,512,768,768/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}"
'''
  
'''
# Generate samples and evaluate on those images

export OUTPUT_DIR="re-evaluations/SD_Nanov21"
export RESULTS_DIR="re-evaluations/SD_Nanov21/samples"
export PRETRAINED_BASE="bguisard/stable-diffusion-nano-2-1"
export MODEL_TYPE="StableDiffusion"
export NUM_SAMPLES=25
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --pretrained_model_name_or_path=$PRETRAINED_BASE \
  --model_type=$MODEL_TYPE \
  --generate_samples=$NUM_SAMPLES \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}"
  #--lora_weights
'''

'''
# Usamos --measure_from_txt y --dont_write para replotear rápido

export OUTPUT_DIR="re-evaluations/LORA_SD_v14"
export RESULTS_DIR="../experimentos2/generic_finetunings_old1/StableDiffusionLoRAv14_batch64_lr1e-04(cosine)_epochs20_notsaved/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --measure_from_txt="${OUTPUT_DIR}/re-evaluation.txt" \
  --dont_write

export OUTPUT_DIR="re-evaluations/LORA_SD_v20"
export RESULTS_DIR="../experimentos2/generic_finetunings_old1/StableDiffusionLoRAv20_batch32_lr1e-04(cosine)_epochs20_notsaved/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --measure_from_txt="${OUTPUT_DIR}/re-evaluation.txt" \
  --dont_write

export OUTPUT_DIR="re-evaluations/LORA_SD_Nanov21_bs512"
export RESULTS_DIR="generic_finetunings_4atts/StableDiffusionLoRA_batch512_lr-cosine-1e-04_epochs20_snrgamma5.0/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --measure_from_txt="${OUTPUT_DIR}/re-evaluation.txt" \
  --dont_write
  
  
export OUTPUT_DIR="re-evaluations/LORA_SD_Nanov21_bs1"
export RESULTS_DIR="generic_finetunings_4atts/StableDiffusionLoRA_batch1*1_lr-cosine-1e-04_epochs20_snrgamma5.0/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --measure_from_txt="${OUTPUT_DIR}/re-evaluation.txt" \
  --dont_write
  
  
export OUTPUT_DIR="re-evaluations/UNet_small"
export RESULTS_DIR="../experimentos2/generic_finetunings_old2_4atts/MyDiffusionSmall_batch3_lr1e-05(cosine)_epochs3_snrgamma5.0/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --measure_from_txt="${OUTPUT_DIR}/re-evaluation.txt" \
  --dont_write
  
export OUTPUT_DIR="re-evaluations/UNet_mid"
export RESULTS_DIR="../experimentos2/generic_finetunings_old2_4atts/MyDiffusionMedium_batch2_lr1e-04(cosine)_epochs3/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --measure_from_txt="${OUTPUT_DIR}/re-evaluation.txt" \
  --dont_write
  
export OUTPUT_DIR="re-evaluations/UNet_big"
export RESULTS_DIR="../experimentos2/generic_finetunings_old2_4atts/MyDiffusion_batch1_lr1e-05(cosine)_epochs3_snrgamma5.0/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --measure_from_txt="${OUTPUT_DIR}/re-evaluation.txt" \
  --dont_write
'''

'''
# One experiment
export OUTPUT_DIR="re-evaluations/experiment"
export RESULTS_DIR="generic_finetunings_4atts/StableDiffusionLoRA_batch64*1_lr-cosine-1e-04_epochs5_snrgamma5.0/validations"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \

# One epoch
export OUTPUT_DIR="re-evaluations/epoch"
export RESULTS_DIR="generic_finetunings_4atts/StableDiffusionLoRA_batch1*1_lr-cosine-1e-04_epochs20_snrgamma5.0/validations/StableDiffusionLoRA_epoch0"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --results_dir="${RESULTS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \

# Ground truth
export OUTPUT_DIR="re-evaluations/ground_truth"
python re-evaluator.py \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=64 \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --output_dir="${OUTPUT_DIR}" \
'''
