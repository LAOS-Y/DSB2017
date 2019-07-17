set -e

cd detector
eps=100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 32 --epochs $eps --save-dir res18 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 32 --resume results/res18/$eps.ckpt --test 1 
cp results/res18/$eps.ckpt ../../model/detector.ckpt
