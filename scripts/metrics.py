from init import Options
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from skimage import measure

from datetime import datetime
import matplotlib.pyplot as plt

from monai.networks.nets import UNETR, UNet, SegResNet,AttentionUnet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import (CacheDataset, 
                DataLoader, decollate_batch, load_decathlon_datalist)
from monai.metrics import (DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric)

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureType,   
    NormalizeIntensityd,
    ScaleIntensityd,)
import pickle 

opt = Options().parser()
def get_tumor_size(label, list_of_sizes):
    label_ = measure.label(label, background=0)
    props = measure.regionprops(label_)
    for prop in props:
        bbox = prop.bbox
        max_diam = np.max([bbox[3] - bbox[0],
                        bbox[4] - bbox[1],
                        bbox[5] - bbox[2]])
        list_of_sizes.append(max_diam)
    return list_of_sizes

def get_nb_tumor(label):
    label = measure.label(label, background=0)
    return np.max(label)

def main():
    datasets = opt.dataset_folder
    val_files = load_decathlon_datalist(datasets, True, "validation")

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim= opt.spacing, mode=("bilinear", "nearest")),
            
            NormalizeIntensityd(keys=['image']),                                          # augmentation
            ScaleIntensityd(keys=['image']), 
            
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=0.0, num_workers=opt.workers)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset {opt.dataset_folder[-12:]} on model {opt.network}")
    device = torch.device("cuda:0")

    if opt.network == 'unetr':
        model = UNETR(
        in_channels=opt.in_channels,
        out_channels=opt.out_channels,
        img_size= opt.patch_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        )
    
    if opt.network == 'UNet':
        model = UNet(
            spatial_dims=3,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    
    if opt.network == "SegResNet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            dropout_prob=0.2,
        )
    
    if opt.network == "Attention":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )

    if opt.test_pretrain is not None:
        model.load_state_dict(torch.load(opt.test_pretrain))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model {opt.network} pretrained loaded")
    
    model.eval()
    model.to(device)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    HD_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile = 95)
    confuse_metric = ConfusionMatrixMetric(include_background=False, 
            metric_name=["f1_score", "accuracy", "recall", "specificity", "precision", "prevalence threshold"],
            compute_sample=False)
    
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    
    metrics = {"dice" : [], "HD" : [],"f1_score" : [], "accuracy" : [], "recall" : [],
        "specificity" : [], "precision" : [], "prevalence" : []}

    slice_map = {'0': 201,'1': 117,'2': 134,'3': 48,'4': 77,'5': 73,'6': 72,'7': 196,'8': 73,
        '9': 147,'10': 119,'11': 80,'12': 32,'13': 48,'14': 202,'15': 208,'16': 46,'17': 195,
        '18': 136,'19': 153,'20': 160,'21': 86,'22': 72,'23': 30,'24': 184,'25': 72,'26': 116,
        '27': 74,
        '28': 97,
        '29': 52,
        '30': 200,
        '31': 66,
        '32': 139,
        '33': 107,
        '34': 89,
        '35': 220,
        '36': 192,
        '37': 59,
        '38': 96,
        '39': 126,
        '40': 159,
        '41': 132,
        '42': 184,
        '43': 217,
        '44': 116,
        '45': 183,
        '46': 160,
        '47': 128,
        '48': 132,
        '49': 165,
        '50': 149,
        '51': 160,
        '52': 219,
        '53': 170}
    dice_size = {}
    nb_tumor = 0

    with torch.no_grad():
        for case_num in range(len(slice_map)):
            img_name = val_files[case_num]['image']
            print(f"{case_num}/{len(slice_map)} Inference on case {img_name}")
            img = val_ds[case_num]["image"]
            label = val_ds[case_num]["label"]
            nb_tumor += get_nb_tumor(label[0])
            val_inputs = torch.unsqueeze(img, 1).cuda()
            val_labels = torch.unsqueeze(label, 1).cuda()
            val_outputs = sliding_window_inference(
                val_inputs, (128, 128, 128), 4, model, overlap=0.8
            )

            val_outputs_ = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels_ = [post_label(i) for i in decollate_batch(val_labels)]
            # compute dice metric for current iteration
            dice_metric(y_pred=val_outputs_, y=val_labels_)
            dice= dice_metric.aggregate().item()
            dice_metric.reset()
            metrics['dice'].append(dice)
            # compute hausdorff metric for current iteration
            HD_metric(y_pred=val_outputs_, y=val_labels_)
            HD = HD_metric.aggregate().item()
            HD_metric.reset()
            metrics['HD'].append(HD)

            # compute confusion matrix for current iteration
            confuse_metric(y_pred=val_outputs_, y=val_labels_)
            f1_score, accuracy, recall, specificity, precision, prevalence  = confuse_metric.aggregate()
            confuse_metric.reset()
            metrics['f1_score'].append(float(f1_score))
            metrics['accuracy'].append(float(accuracy))
            metrics['recall'].append(float(recall))
            metrics['specificity'].append(float(specificity))
            metrics['precision'].append(float(precision))
            metrics['prevalence'].append(float(prevalence))

    print("-----------------------------------------------------")
    print(f"Number of Tumors : {nb_tumor}\n")
    for metric in metrics.keys():
        print(f"Mean {metric} : {np.nanmean(metrics[metric]):.3f} (std : {np.nanstd(metrics[metric]):.3f})")

if __name__ == "__main__":
    main()
