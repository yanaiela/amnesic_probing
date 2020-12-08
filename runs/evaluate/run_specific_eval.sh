AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

vecs=$1
labels=$2
text=$3
deprobe_dir=$4
device=$5

python amnesic_probing/tasks/task_specific_eval.py \
        --vecs $vecs \
        --labels $labels \
        --text $text \
        --deprobe_dir $deprobe_dir \
        --device $device \
        --wandb
