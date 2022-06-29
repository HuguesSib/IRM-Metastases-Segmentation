import argparse
from email.policy import default 
import os 

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        #Some parameters
        parser.add_argument('--dataset_folder', type = str, 
                                default = "../dataset/dataset_1.json",
                                help = "JSON dataset containing both training and validation set")
        parser.add_argument('--model_folder', type = str,
                                default = "/itet-stor/hsibille/bmicdatasets_bmicnas01/Processed/Hugues_Sibille")
        parser.add_argument("--workers", default = 16, type = int,
                                help = 'number of data loading workers')
        parser.add_argument("--gpus", default = '0', type = str,
                                help = 'Number of available GPUs, e.g 0 0,1,2 0,2 use -1 for CPU')
        parser.add_argument("--pretrain", 
                    default = None, 
                    type = str, 
                    help = '/itet-stor/hsibille/bmicdatasets_bmicnas01/Processed/Hugues_Sibille/UNETR_1150epoch.pth')
        parser.add_argument("--test_pretrain",
                    default = '/itet-stor/hsibille/bmicdatasets_bmicnas01/Processed/Hugues_Sibille/UNETR_1400epoch.pth', 
                    type = str, 
                    help = '/itet-stor/hsibille/bmicdatasets_bmicnas01/Processed/Hugues_Sibille/best_metric_model.pth')
        #Model parameter
        parser.add_argument("--network", default = 'unetr', help = 'Add any net ("UNet" to try), or "ViT" ')
        parser.add_argument("--patch_size", default = (128, 128, 128), help = "Size of the patches extracted from the image")
        parser.add_argument("--spacing", default = [1.0, 1.0, 1.0], help  = "Resolution of the MRI")

        parser.add_argument("--batch_size", type = int, default = 2)
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=2, type=int, help='Channels of the output')

        #Training parameters
        parser.add_argument('--epochs',default = 350, type = int)
        parser.add_argument('--lr', default = 0.0001, help = "Learning rate")

        #Inference
        #####TO-DO
        #parser.add_argument('--metric', default = 'dice', type =str, help = "Can also be HD (Hausdorff distance)")
        self.initialized = True
        return parser
    
    def parser(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()

        #Set gpus ids
        if opt.gpus != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        
        return opt

