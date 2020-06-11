

SimCLR:

```

CUDA_VISIBLE_DEVICES=1 python run.py --train_mode=pretrain   --train_batch_size=512 --train_epochs=1000   --learning_rate=1.0 --weight_decay=1e-6 --temperature=0.5   --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18   --use_blur=False --color_jitter_strength=0.5   --model_dir=checkpoint/ --use_tpu=False


CUDA_VISIBLE_DEVICES=2 python run.py --train_mode=pretrain   --train_batch_size=512 --train_epochs=1000   --learning_rate=1.0 --weight_decay=1e-6 --temperature=0.5   --dataset=stl10 --image_size=48 --eval_split=test --resnet_depth=34   --use_blur=False --color_jitter_strength=0.5   --model_dir=checkpoint_stl10/ --use_tpu=False


CUDA_VISIBLE_DEVICES=1 python run.py --mode=train_then_eval --train_mode=finetune --fine_tune_after_block=4 --zero_init_logits_layer=True --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 --checkpoint=checkpoint/ --model_dir=simclr_test_ft --use_tpu=False


CUDA_VISIBLE_DEVICES=2 python run.py --mode=train_then_eval --train_mode=finetune --fine_tune_after_block=4 --zero_init_logits_layer=True --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 --dataset=stl10 --image_size=48 --eval_split=test --resnet_depth=34 --checkpoint=checkpoint_stl10/ --model_dir=simclr_stl10_ft --use_tpu=False

```

```
64x2 + 3x16 = 176

real: 64, real_rotated: 24, fake: 64, fake_rotated: 24
```



cifar-10:

```
58
================False + conditional_batch_norm + 2-layer E

c8bbf50b08b8, compare_gan_2

CUDA_VISIBLE_DEVICES=11 python compare_gan/main.py --gin_config example_configs/s3gan_cifar.gin &

CUDA_VISIBLE_DEVICES=7 python compare_gan/main.py --gin_config example_configs/s3gan_cifar.gin --schedule=continuous_eval --eval_every_steps=0


60
================False + conditional_batch_norm + 2-layer E

bc1929454d21, compare_gan_cifar100_2

CUDA_VISIBLE_DEVICES=15 python compare_gan/main.py --gin_config example_configs/s3gan_cifar.gin &

CUDA_VISIBLE_DEVICES=3 python compare_gan/main.py --gin_config example_configs/s3gan_cifar.gin --schedule=continuous_eval --eval_every_steps=0
```

-----------------------------------------------------------------------------------------


cifar100-20:

```
================False + conditional_batch_norm + 3-layer E

30d9dbddcf2c, compare_gan_cifar100_3

CUDA_VISIBLE_DEVICES=7 python compare_gan/main.py --gin_config example_configs/s3gan_cifar100.gin &

CUDA_VISIBLE_DEVICES=3 python compare_gan/main.py --gin_config example_configs/s3gan_cifar100.gin --schedule=continuous_eval --eval_every_steps=0

```

-----------------------------------------------------------------------------------------


stl-10:

```
================10 + False + conditional_batch_norm + 2-layer E

6dd52a2c77a1, compare_gan_stl10_5

CUDA_VISIBLE_DEVICES=8 python compare_gan/main.py --gin_config example_configs/s3gan_stl.gin &

CUDA_VISIBLE_DEVICES=3 python compare_gan/main.py --gin_config example_configs/s3gan_stl.gin --schedule=continuous_eval --eval_every_steps=0
```

-----------------------------------------------------------------------------------------
