
export OUTPUT_DIR="samples"
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7.5

export PRETRAINED_IF_NEEDED="bguisard/stable-diffusion-nano-2-1"
export MAIN_FOLDER="../RUNPOD/generic_finetunings_4atts"
export EXPERIMENT="DDPM_batch256*1_lr-cosine-1e-04_epochs10_snrgamma5.0"
export MODEL_TYPE="DDPM"
export UNET_WEIGHTS="../RUNPOD/generic_finetunings_4atts/DDPM_batch256*1_lr-cosine-1e-04_epochs10_snrgamma5.0/unet.pth.tar"
python sampler.py \
  --main_folder="${MAIN_FOLDER}" \
  --experiment="${EXPERIMENT}" \
  --model_type="${MODEL_TYPE}" \
  --num_inference_steps=$NUM_INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --output_dir="${OUTPUT_DIR}" \
  --pretrained_model_name_or_path="${PRETRAINED_IF_NEEDED}" \
  --unet_weights="${UNET_WEIGHTS}"

'''
export PRETRAINED_IF_NEEDED="bguisard/stable-diffusion-nano-2-1"
export MAIN_FOLDER="generic_finetunings_4atts"
export EXPERIMENT="StableDiffusionLoRA_batch1*1_lr-cosine-1e-04_epochs20_snrgamma5.0"
export MODEL_TYPE="StableDiffusionLoRA"
python sampler.py \
  --main_folder="${MAIN_FOLDER}" \
  --experiment="${EXPERIMENT}" \
  --model_type="${MODEL_TYPE}" \
  --num_inference_steps=$NUM_INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --output_dir="${OUTPUT_DIR}" \
  --pretrained_model_name_or_path="${PRETRAINED_IF_NEEDED}" \

export PRETRAINED_IF_NEEDED="../experimentos2/generic_finetunings_old2_4atts/MyDiffusion_batch1_lr1e-05(cosine)_epochs3_snrgamma5.0"
export MAIN_FOLDER="../experimentos2/generic_finetunings_old2_4atts"
export EXPERIMENT="MyDiffusion_batch1_lr1e-05(cosine)_epochs3_snrgamma5.0"
export MODEL_TYPE="MyDiffusion"
python sampler.py \
  --main_folder="${MAIN_FOLDER}" \
  --experiment="${EXPERIMENT}" \
  --model_type="${MODEL_TYPE}" \
  --num_inference_steps=$NUM_INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --output_dir="${OUTPUT_DIR}" \
  --pretrained_model_name_or_path="${PRETRAINED_IF_NEEDED}" \
'''
