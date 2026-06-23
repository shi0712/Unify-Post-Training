set -x
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)

ray stop
# ------------------------------------------------------------------------------------------------
export PYTHONPATH=$ROOT:$PYTHONPATH

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

source activate uft
# ------------------------------------------------------------------------------------------------
# NOTE: change to your root dir
ROOT=/mnt/petrelfs/languanzhou/sjw/Unify-Post-Training

# export SWANLAB_API_KEY='16xw3bqnlqJYli0MXham3'
# export SWANLAB_DESCRIPTION='Pure SFT with phi-function weighting'
export WANDB_API_KEY='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'
export WANDB_PROJECT="unified-ft"

OFFLINE_LOSS_TYPE="pure_sft"
SFT_LOSS_COEF=1.0

LR=5e-6
MODEL=Qwen2.5-Math-1.5B
EXP_NAME="${DATE}_${OFFLINE_LOSS_TYPE}_${MODEL}_lr@${LR}_${TIME_TAG}"
MODEL_PATH=/mnt/inspurfs/evla2_t/pretrained/Qwen/Qwen2.5-Math-1.5B
DATA_DIR=$ROOT/data/

cd $ROOT/hpt/verl/
mkdir -p /mnt/inspurfs/evla2_t/languanzhou/ckpts/$EXP_NAME

TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/openr1.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_DIR}/AIME24/test.parquet","${DATA_DIR}/AIME25/test.parquet","${DATA_DIR}/AMC23/test.parquet","${DATA_DIR}/MATH-500/test.parquet","${DATA_DIR}/Minerva/test.parquet","${DATA_DIR}/Olympiad-Bench/test.parquet"]}

python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.reward_impl_version=6 \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.offline_loss_type="$OFFLINE_LOSS_TYPE" \
    actor_rollout_ref.actor.sft_loss_coef=$SFT_LOSS_COEF \
    actor_rollout_ref.actor.enable_phi_function=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    +actor_rollout_ref.rollout.val_top_p=0.95 \
    actor_rollout_ref.rollout.n_val=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.unify_strategy=no \
    trainer.max_optim_to_keep=2 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=/mnt/inspurfs/evla2_t/languanzhou/ckpts/$EXP_NAME
