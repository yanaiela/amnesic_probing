AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

proj_vecs=$1
labels=$2
text=$3
task=$4
device=$5

python amnesic_probing/tasks/layer_wise_lm.py \
        --proj_vecs $proj_vecs \
        --labels $labels \
        --text $text \
        --task $task \
        --n 100000 \
        --device $device \
        --wandb
