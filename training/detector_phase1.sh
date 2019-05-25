set -e

cd detector
eps=100
CUDA_VISIBLE_DEVICES=0,7 python main.py --model res18 -b 8 --epochs $eps --save-dir res18 -j 8 #> ../detector_phase1.txt 
