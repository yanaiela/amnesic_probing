#!/bin/bash

TAGS=( ORG CARDINAL GPE DATE PERSON )
TAGS_NAME=( ORG CARDINAL GPE DATE PERSON )

for encode_format in normal masked
do
  for ((i=0;i<${#TAGS[@]};++i))
  do
    for split in train dev
    do
      echo "data/ontonotes_output_$encode_format/$split/ner_"${TAGS_NAME[i]}""
      python amnesic_probing/tasks/reduce_labels.py \
            --label data/ontonotes_output_$encode_format/$split/ner.pickle \
            --keep_label "${TAGS[i]}" \
            --out_labels data/ontonotes_output_$encode_format/$split/ner_"${TAGS_NAME[i]}".pickle
    done
  done
done
