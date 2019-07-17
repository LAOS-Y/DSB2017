set -e

cd detector
eps=100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7,8 python main.py --model res18 -b 32 --resume ../../model/detector.ckpt --test 1 -j 32 --n_test  8 --epochs 5
#cp results/res18/$eps.ckpt ../../model/detector.ckpt
