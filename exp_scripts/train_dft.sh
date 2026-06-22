set -x
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)

# ------------------------------------------------------------------------------------------------
export PYTHONPATH=$ROOT:$PYTHONPATH

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

source activate uft
# ------------------------------------------------------------------------------------------------
# NOTE: change to your root dir
ROOT=../Unify-Post-Training

export SWANLAB_API_KEY='16xw3bqnlqJYli0MXham3'
export SWANLAB_DESCRIPTION='SFT with phi-function per-token probability weighting'
export WANDB_PROJECT="unified-ft"

LR=1e-6
MODEL=Qwen2.5-Math-1.5B
EXP_NAME="${DATE}_sft-dft_${MODEL}_lr@${LR}_${TIME_TAG}"
MODEL_PATH=$ROOT/models/$MODEL
DATA_DIR=$ROOT/data/

cd $ROOT/hpt/verl/
mkdir -p $ROOT/checkpoints/$EXP_NAME

TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_DIR}/AIME24/test.parquet","${DATA_DIR}/AMC23/test.parquet","${DATA_DIR}/MATH-500/test.parquet"]}

python3 -m verl.trainer.fsdp_dft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=128 \
    data.micro_batch_size=16 \
    data.max_length=1024 \
    data.truncation=error \
    data.balance_dp_token=False \
    data.prompt_key=prompt \
    data.response_key=response \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    optim.lr=$LR \
    optim.betas="[0.9,0.95]" \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    trainer.total_epochs=4 \
    trainer.total_training_steps=500 \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.logger="['console','swanlab']" \
    trainer.default_local_dir=$ROOT/checkpoints/$EXP_NAME
