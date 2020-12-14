#! /usr/bin/bash
device=4
cmd="/home/jingyi/Python-3.6.7/bin/python3 generate.py 
data-bin/iwslt14.tokenized.de-en --path 
/home/jingyi/fairseq-research/checkpoints/iwslt-de2en/12_3_connEn_De_gate_topAndCurrent_Norm_head8_12/checkpoint_best.pt 
--remove-bpe 
--batch-size 64
--beam 8 --output temp.txt
"
export CUDA_VISIBLE_DEVICES=$device
echo $cmd
eval $cmd
