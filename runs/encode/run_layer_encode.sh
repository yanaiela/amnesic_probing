AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

input_file=$1
projections_dir=$2
output_dir=$3
task_format=$4
encode_format=$5
control=$6
device=$7


python amnesic_probing/encoders/encode_with_forward_pass/encode.py \
        --input_file $input_file  \
        --projections_dir $projections_dir \
        --output_dir $output_dir/ \
        --encoder bert-base-uncased \
        --format $task_format \
        --encode_format $encode_format \
        --control $control \
        --device $device \
        --num_layers 12
