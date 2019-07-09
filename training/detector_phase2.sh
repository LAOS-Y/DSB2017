set -e

cd detector
eps=100
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py --model res18 -b 32 --resume results/res18/$eps.ckpt --test 1 -j 32 --n_test 1
cp results/res18/$eps.ckpt ../../model/detector.ckpt
