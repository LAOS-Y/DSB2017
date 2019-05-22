set -e

cd detector
eps=30
CUDA_VISIBLE_DEVICES=0 python main.py --model res18 -b 8 --epochs $eps --save-dir res18 -j 8 #> ../detector_phase1.txt 
