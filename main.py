from preprocessing import full_prep
from config_submit import config as config_submit

import torch
#torch.backends.cudnn.enabled = False

from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas

datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if not skip_prep:
    print("start preprocessing")
    testsplit = full_prep(datapath,prep_result_path,
                          n_worker = config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
    print("finish preprocessing")
else:
    print("skip preprocessing")
    testsplit = os.listdir(prep_result_path)
    testsplit = [i.split('_')[0] for i in testsplit if "clean.npy" in i]

#print(testsplit)
#print(len(testsplit))
#0/0

print("start building detector")

nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

print("finish loading detector param")

torch.cuda.set_device(0)
nod_net = nod_net.cuda()
cudnn.benchmark = True
nod_net = DataParallel(nod_net)
#nod_net = nod_net.cuda()

bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

    print("start building dataset for detection")

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)


    print("start building dataloader for detection")

    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)

    print("start detection")
    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])
    print("finish detection")
else:
    print("skip detection")


#0/0

print("start building casenet")

casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
casenet = casemodel.CaseNet(topk=5)
config2 = casemodel.config
#checkpoint = torch.load(config_submit['classifier_param'])
#`casenet.load_state_dict(checkpoint['state_dict'])

print("finish loading casenet param")

torch.cuda.set_device(0)
casenet = casenet.cuda()
cudnn.benchmark = True
casenet = DataParallel(casenet)

filename = config_submit['outputfile']



def test_casenet(model,testset):
    print("start building dataloader for casenet")

    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 32,
        pin_memory=True)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()

    from tqdm import tqdm
    for i,(x,coord) in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            coord = Variable(coord).cuda()
            x = Variable(x).cuda()
            nodulePred,casePred,_ = model(x,coord)
            predlist.append(casePred.data.cpu().numpy())
            #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])

    predlist = np.concatenate(predlist)
    return predlist    

config2['bboxpath'] = bbox_result_path
config2['datadir'] = prep_result_path

print("start building dataset for casenet")

dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
predlist = test_casenet(casenet,dataset).T

print("finisht casenet inference")

import ipdb
ipdb.set_trace()

df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
df.to_csv(filename,index=False)

print("csv saved as ", filename)
