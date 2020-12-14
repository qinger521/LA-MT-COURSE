#! /usr/bin/bash
set -e


device=4
encoder_layers=6
task=wmt-en2de


# must set this tag
#tag=base-reg6-2_qkv_mean
tag=wmt_basline
if [ $task == "wmt-en2de" ]; then
        arch=relative_transformer_t2t_wmt_en_de
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.002
        warmup=16000
        max_tokens=4096
        update_freq=2
        weight_decay=0.0
        keep_last_epochs=20
        max_epoch=21
        max_update=
        data_dir=wmt_17
        src_lang=en
        tgt_lang=de
elif [ $task == "iwslt-de2en" ]; then
        arch=heterogeneous_dense_transformer_iwslt_de_en
        share_embedding=1
        share_decoder_input_output_embed=0
 	criterion=label_smoothed_cross_entropy
        fp16=0
        lr=0.0015
        warmup=8000
        max_tokens=4096
        update_freq=1
        weight_decay=0.0001
        keep_last_epochs=10
        max_epoch=51
        max_update=
        data_dir=iwslt14.tokenized.de-en
        src_lang=de
        tgt_lang=en
else
        echo "unknown task=$task"
        exit
fi

save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="/home/jingyi/Python-3.6.7/bin/python3 -u train.py data-bin/$data_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --weight-decay $weight_decay
  --encoder-layers $encoder_layers
  --criterion $criterion  --label-smoothing 0.1
  --max-tokens $max_tokens
  --update-freq $update_freq
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --seed 1 
  --save-dir $save_dir
  --keep-last-epochs $keep_last_epochs
  --tensorboard-logdir $save_dir"

adam_betas="'(0.9, 0.98)'"
cmd=${cmd}" --adam-betas "${adam_betas}
if [ $share_embedding -eq 1 ]; then
cmd=${cmd}" --share-all-embeddings "
fi
if [ $share_decoder_input_output_embed -eq 1 ]; then
cmd=${cmd}" --share-decoder-input-output-embed "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi
if [ -n "$dropout" ]; then
cmd=${cmd}" --dropout "${dropout}
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

#echo $cmd
#eval $cmd
#cmd=$(eval $cmd)
#nohup $cmd exec 1> $save_dir/train.log exec 2>&1 &
#tail -f $save_dir/train.log

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
