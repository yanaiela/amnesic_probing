AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

vecs=$1
labels=$2
text=$3
deprobe_dir=$4
task_type=$5
device=$6

python amnesic_probing/tasks/lm_per_dim.py \
        --vecs $vecs \
        --labels $labels \
        --text $text \
        --task $task_type \
        --deprobe_dir $deprobe_dir \
        --display_examples 100 \
        --device $device \
        --wandb
