AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

layers=$1
proj_vecs=$2
labels=$3
text=$4
task=$5

python amnesic_probing/tasks/layer_wise_deprobe.py \
        --layers $layers \
        --proj_vecs $proj_vecs \
        --labels $labels \
        --text $text \
        --task $task \
        --wandb
