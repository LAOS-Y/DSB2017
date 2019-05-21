set -e

cd detector
eps=30
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 16 --epochs $eps --save-dir res18 -j 16 #> ../detector_phase1.txt 
