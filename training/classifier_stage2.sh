cd ./classifier
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --model1  net_detector_3 --model2  net_classifier_3 -b 20 -b2 12 --save-dir net3 --resume ./results/start.ckpt --start-epoch 30 --epochs 130 -j 1
