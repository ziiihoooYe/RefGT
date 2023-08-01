python3 ./train/train_denoise.py --arch Uformer_B --batch_size 32 --gpu '0,1, 2, 3, 4, 5, 6, 7' \
    --train_ps 128 --train_dir ../../../data/BDD100K/train --env _0706 \
    --val_dir ../../../data/BDD100K/val --save_dir ./logs/ \
    --dataset sidd --warmup 