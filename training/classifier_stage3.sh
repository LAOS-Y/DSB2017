cd ./classifier
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --model1  net_detector_3 --model2  net_classifier_4 -b 24 -b2 12 --save-dir net4 --resume ./results/net3/125.ckpt --freeze_batchnorm 1 --start-epoch 121 --lr 0.001
