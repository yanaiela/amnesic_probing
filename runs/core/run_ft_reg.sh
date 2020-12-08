AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

train_path=$1
out_dir=$2
debias_p=$3

python amnesic_probing/debiased_finetuning/debiased_finetuning_lm.py \
        --train_path $train_path \
        --out_dir $out_dir \
        --n_epochs 20 \
        --debias $debias_p \
        --device cuda:0 \
        --wandb
