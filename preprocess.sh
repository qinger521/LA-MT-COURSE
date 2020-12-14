src=en
tgt=de
TEXT=data-bin/wmt
tag=wmt_17
output=data-bin/$tag
srcdict=$TEXT/dict.$src.txt
tgtdict=$TEXT/dict.$tgt.txt

python3 preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/train  --validpref $TEXT/valid --testpref $TEXT/test --destdir $output --workers 32
