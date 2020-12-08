#!/bin/bash

#TAGS=( VERB NOUN ADP DET NUM . PRT CONJ ADV PRON ADJ X )
#TAGS_NAME=( verb noun adp det num punct prt conj adv pron adj other )
TAGS_NAME=(adpmod det compmod num adpobj p poss adp amod nsubj dep dobj cc conj advmod ROOT ccomp aux xcomp neg)
TAGS=(adpmod det compmod num adpobj p poss adp amod nsubj dep dobj cc conj advmod ROOT ccomp aux xcomp neg)

for encode_format in normal masked
do
  for ((i=0;i<${#TAGS[@]};++i))
  do
    for split in train dev
    do
      python amnesic_probing/tasks/reduce_labels.py \
            --label data/ud_output_$encode_format/$split/dep.pickle \
            --keep_label "${TAGS[i]}" \
            --out_labels data/ud_output_$encode_format/$split/dep_"${TAGS_NAME[i]}".pickle
    done
  done
done
