AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

vecs=$1
labels=$2
out_dir=$3
task=$4
balance=$5

python amnesic_probing/tasks/remove_property.py \
        --vecs $vecs \
        --labels $labels \
        --out_dir $out_dir \
        --n_cls 100 \
        --task $task \
        --input_dim 768 \
        --balance_data $balance \
        --wandb
