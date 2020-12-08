AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

train_path=$1
debias_p=$2
rebias=$3
out_dir=$4

python amnesic_probing/debiased_finetuning/rebiased_finetuning_lm.py \
        --train_path $train_path \
        --debias $debias_p \
        --rebias $rebias \
        --n_epochs 20 \
        --out_dir $out_dir \
        --device cuda:0 \
        --wandb
