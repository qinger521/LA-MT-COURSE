#! /usr/bin/bash
set -e
echo "jingyi_train"
task=wmt-en2de
tag=base20-reg6_qk+1
arch=transformer_iwslt_de_en
data_dir=iwslt14.tokenized.de-en
src_lang=de
tgt_lang=en

save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
#cp ${BASH_SOURCE[0]} $save_dir/train.sh

#gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="python3 -u j_train.py data-bin/$data_dir
   -s $src_lang -t $tgt_lang
  --arch $arch
  --max-tokens 4000
  --optimizer depth_scale_adam --clip-norm 0.0
  --depth-scale 1
  --num-encoder-layers 6
  --lr-scheduler inverse_sqrt
  --tensorboard-logdir $save_dir"

# --depth-scale 1
#  --num-encoder-layers 6
# python3 -u j_train.py data-bin/iwslt14.tokenized.de-en -s de
# -t en --arch transformer_iwslt_de_en --max-tokens 4000  --optimizer adam
# --clip-norm 0.0  --lr-scheduler inverse_sqrt
echo $cmd
eval $cmd
tail -f $save_dir/train.log
