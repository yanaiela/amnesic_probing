AMNESIC_PATH="PATH-TO-AMNESIC-PROBING"
cd $AMNESIC_PATH
export PYTHONPATH=$AMNESIC_PATH

split=$1
input_file=$2
output_dir=$3
task_format=$4
encode_format=$5
device=$6


python amnesic_probing/encoders/encode.py \
        --input_file $input_file  \
        --output_dir $output_dir/$split/ \
        --encoder bert-base-uncased \
        --format $task_format \
        --encode_format $encode_format \
        --all_layers \
        --device $device
