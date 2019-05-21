set -e

cd detector
eps=100
CUDA_VISIBLE_DEVICES=4 python main.py --model res18 -b 32 --epochs $eps --save-dir res18 -j 32 #> ../detector_phase1.txt 
