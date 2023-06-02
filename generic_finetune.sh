export DATA_DIR="../experimentos2/data/dataset_4atts"
export OUTPUT_DIR="generic_finetunings_4atts"
export EPOCHS=20
export VAL_SAMPLINGS=25
export CHKPTS_EVERY_EPOCHS=1
export ATTESTIMATOR_CHECKPOINT="/home/jorge/UPV/ARF/experimentos2/face_attribute_estimator_output/checkpoint/checkpoint.pth.tar"

'''
export MODEL_TYPE="Nano21StableDiffusion"
export BATCH_SIZE=4
export LR_SCHED="cosine"
export LR=1e-03
python generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lr${LR}(${LR_SCHED})_epochs${EPOCHS}_noaccelerator" \
  #--gradient_accumulation_steps=1 \
  #--resume_from_checkpoint="latest" \
'''
'''
export MODEL_TYPE="NanoStableDiffusion"
export BATCH_SIZE=1024
export LR_SCHED="cosine"
export LR=1e-04
python generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lr${LR}(${LR_SCHED})_epochs${EPOCHS}_noaccelerator" \
  #--gradient_accumulation_steps=1 \
  #--resume_from_checkpoint="latest" \
  
export MODEL_TYPE="MiniStableDiffusion"
export BATCH_SIZE=128
export LR_SCHED="constant"
export LR=1e-04
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lr${LR}(${LR_SCHED})_epochs${EPOCHS}" \
  #--gradient_checkpointing \
  #--gradient_accumulation_steps=4 \
  #--resume_from_checkpoint="latest" \

for i in $(echo $SUDO_PASS | sudo -S lsof /dev/nvidia0 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done

export MODEL_TYPE="StableDiffusion"
export BATCH_SIZE=1
export LR_SCHED="constant"
export LR=1e-04
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS  \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lr${LR}(${LR_SCHED})_epochs${EPOCHS}" \
  #--gradient_checkpointing \
  #--gradient_accumulation_steps=4 \
  #--resume_from_checkpoint="latest" \

export MODEL_TYPE="StableDiffusionLoRA"
export BATCH_SIZE=1
export LR_SCHED="cosine"
export LR=1e-04
export SNR_GAMMA=5.0
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS  \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --snr_gamma=$SNR_GAMMA \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lr-${LR_SCHED}-${LR}_epochs${EPOCHS}_snrgamma${SNR_GAMMA}_TEMPORAL" \
  #--gradient_checkpointing \
  #--gradient_accumulation_steps=4 \
  #--resume_from_checkpoint="latest" \

export BATCH_SIZE=1
export MODEL_TYPE="DeepFloydIF"
export LR_SCHED="cosine"
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="DeepFloyd/IF-I-M-v1.0" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS  \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lrsch-${LR_SCHED}_epochs${EPOCHS}" \
  #--resume_from_checkpoint="latest" \

export BATCH_SIZE=8
export MODEL_TYPE="NanoDeepFloydIF"
export LR_SCHED="cosine"
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="DeepFloyd/IF-I-M-v1.0" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=8 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS  \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lrsch-${LR_SCHED}_epochs${EPOCHS}" \
  #--resume_from_checkpoint="latest" \
 

export BATCH_SIZE=1
export MODEL_TYPE="MiniDeepFloydIF"
export LR_SCHED="cosine"
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="DeepFloyd/IF-I-M-v1.0" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=8 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS  \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lrsch-${LR_SCHED}_epochs${EPOCHS}" \
  #--resume_from_checkpoint="latest" \

export BATCH_SIZE=8
export MODEL_TYPE="DeepFloydIFLoRA"
export LR_SCHED="cosine"
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="DeepFloyd/IF-I-M-v1.0" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lrsch-${LR_SCHED}_epochs${EPOCHS}" \
  #--resume_from_checkpoint="latest" \
  

export BATCH_SIZE=8
export MODEL_TYPE="VQDiffusion"
export LR_SCHED="cosine"
accelerate launch generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="microsoft/vq-diffusion-ithq" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lrsch-${LR_SCHED}_epochs${EPOCHS}" \
  #--resume_from_checkpoint="latest" \

export MODEL_TYPE="MyDiffusion"
export LR_SCHED="cosine"
export BATCH_SIZE=1
export LR=1e-04
export SNR_GAMMA=5.0
export BLOCK_OUT_CHANNELS="256,512,768,768"
export GRAD_ACC=1
python generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --caption_column="text" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --snr_gamma=$SNR_GAMMA \
  --block_out_channels=$BLOCK_OUT_CHANNELS \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --gradient_accumulation_steps=$GRAD_ACC \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}*${GRAD_ACC}_lr-${LR_SCHED}-${LR}_epochs${EPOCHS}_snrgamma${SNR_GAMMA}_channels${BLOCK_OUT_CHANNELS}" \
  #--gradient_accumulation_steps=1 \
  #--resume_from_checkpoint="latest" \

export MODEL_TYPE="DDPM"
export LR_SCHED="cosine"
export BATCH_SIZE=1 #16
export LR=1e-04
export SNR_GAMMA=5.0
export GRAD_ACC=1
python generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --class_column="class" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --snr_gamma=$SNR_GAMMA \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --gradient_accumulation_steps=$GRAD_ACC \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}*${GRAD_ACC}_lr-${LR_SCHED}-${LR}_epochs${EPOCHS}_snrgamma${SNR_GAMMA}" \
  #--pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  #--resume_from_checkpoint="latest" \

export MODEL_TYPE="DiTnoVAE"
export BATCH_SIZE=1
export LR_SCHED="cosine"
export LR=1e-04
python generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="facebook/DiT-XL-2-256" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --class_column="class" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="fp16" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}_lr${LR}(${LR_SCHED})_epochs${EPOCHS}" \
  #--gradient_accumulation_steps=1 \
  #--resume_from_checkpoint="latest" \
'''
'''
export MODEL_TYPE="DDPM"
export LR_SCHED="reduceonplateau"
export BATCH_SIZE=32
export LR=1e-04
export SNR_GAMMA=5.0
export GRAD_ACC=1
python generic_train_text_to_image.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  --data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --image_column="image" \
  --class_column="class" \
  --resolution=64 --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --mixed_precision="no" \
  --num_train_epochs=$EPOCHS \
  --validation_epochs=1 \
  --num_validation_samplings=$VAL_SAMPLINGS \
  --checkpointing_epochs=$CHKPTS_EVERY_EPOCHS \
  --checkpoints_total_limit=10 \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="${LR_SCHED}" --lr_warmup_steps=0 \
  --snr_gamma=$SNR_GAMMA \
  --attestimator_checkpoint=$ATTESTIMATOR_CHECKPOINT \
  --gradient_accumulation_steps=$GRAD_ACC \
  --output_dir="${OUTPUT_DIR}/${MODEL_TYPE}_batch${BATCH_SIZE}*${GRAD_ACC}_lr-${LR_SCHED}-${LR}_epochs${EPOCHS}_snrgamma${SNR_GAMMA}" \
  #--resume_from_checkpoint="latest" \
''' 



