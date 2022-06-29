import os
from xxlimited import new
import matplotlib.pyplot as plt

import numpy as np

import random
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    ScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    Spacingd,
    EnsureTyped,
    SpatialPadd,
    EnsureType,
    RandScaleIntensityd,
    ToTensord,
)
from monai.utils import first
from monai.data.utils import pad_list_data_collate
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, UNet
from monai.networks.layers import Norm

from skimage import measure
from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from datetime import datetime
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from init import Options

print_config()
im_size = 128

def get_ROI_image(img, label, new_dataset,z = True, size = im_size//2):
    global metastases_count
    label_ = measure.label(label, background=0)
    props = measure.regionprops(label_)
    
    for i in range(len(props)):
        if z :
            offset = [random.randint(- size//2, size//2), 
                    random.randint(- size//2, size//2), 
                    random.randint( -size//2, size//2)]
        
        #This is just for visualisation
        else:
            offset = [0, 
                    random.randint(- size//2, size//2), 
                    random.randint(- size//2, size//2)]
        centr = np.floor(props[i].centroid).astype(int) + offset
        bb = (centr[0] - size,
                centr[0] + size,
                centr[1] - size,
                centr[1] + size,
                centr[2] - size,
                centr[2] + size)
        new_image = img[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        new_label = label[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]

        if new_image.shape == (2*size, 2*size, 2*size):
            new_dataset.append(
                {
                'image' : new_image.reshape((1, new_image.shape[0],new_image.shape[1], new_image.shape[2])),
                'label' : new_label.reshape((1, new_label.shape[0],new_label.shape[1], new_label.shape[2]))
                }
            )
    return new_dataset

def plot(filename, loss_dice_plot = {}):
    if len(loss_dice_plot) != 0 :
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss", fontweight = 'bold')
        x = [i for i in range(len(loss_dice_plot['epoch_loss_values']))]
        y = loss_dice_plot['epoch_loss_values']
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid()
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice", fontweight = 'bold')
        x = [(loss_dice_plot['val_interval'] * i)  for i in range(len(loss_dice_plot['metric_values']))]
        y = loss_dice_plot['metric_values']
        plt.xlabel("epoch")
        plt.ylabel("dice score")
        plt.grid()
        plt.plot(x, y)
        plt.savefig('../results/' + filename)
        plt.show()

def train(new_train, new_val):
    opt = Options().parser()

    data_dir = "../dataset/"
    split_JSON = "dataset_1.json"

    datasets = data_dir + split_JSON
    train_files= load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")

    root_dir = opt.model_folder
    save_name = "ROI_unetr_1406"
    if opt.gpus != '-1':
        num_gpus = len(opt.gpus.split(','))
    else:
        num_gpus = 0
    print('Number of GPU :', num_gpus)


    train_transforms = Compose(
        [
        LoadImaged(keys = ['image', 'label']),
        AddChanneld(keys = ["image", 'label']), 
        Orientationd(keys=["image", "label"], axcodes="RAS"),        
        Spacingd(keys=["image", "label"], pixdim= [1.0, 1.0, 1.0], mode=("bilinear", "nearest")),

        NormalizeIntensityd(keys=['image']),     
        ScaleIntensityd(keys=['image']), 

        ### TO-DO : ADD Data augmentation transformation
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys = ['image', 'label']),
            AddChanneld(keys = ["image", 'label']), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim= [1.0, 1.0, 1.0], mode=("bilinear", "nearest")),

            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']), 

            EnsureTyped(keys=["image", "label"]),
        ]
    )

    #Setup Loaders
    ##Cache rate parameters are used to store data in the cache (depends on the memory)
    train_ds= CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=0.0, num_workers= opt.workers)
    total = 80
    train_count, val_count = round(0.8* total), round(0.2*total)
    count = 0
    for data in train_ds:
        get_ROI_image(data['image'][0], data['label'][0], new_train)
        count += 1
        if count == train_count:
            break
        
    train_loader = DataLoader(new_train, batch_size=opt.batch_size,
        shuffle=True, num_workers=opt.workers)

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=0.0, num_workers=opt.workers)

    count = 0
    for data in val_ds:
        get_ROI_image(data['image'][0], data['label'][0], new_val)
        count += 1
        if count == val_count:
            break
        
    val_loader = DataLoader(new_val, batch_size=1,
        shuffle = False, num_workers=opt.workers)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New dataset created")
    print("Number of training samples : ", len(new_train),\
        "\nNumber of validation samples : ", len(new_val))


    device = torch.device("cuda:0") #Change if multiple gpu training

    if opt.network == 'unetr':
        model = UNETR(
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            img_size=(im_size, im_size, im_size),
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

    if num_gpus > 0:
        if num_gpus > 1:
            model.cuda()
            model= torch.nn.DataParallel(model)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model {opt.network} set on multiple gpus.")
        else:
            model.to(device)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model {opt.network} set on GPU")
    else : 
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model {opt.network} set on CPU")

    if opt.pretrain is not None:
        model.load_state_dict(torch.load(opt.pretrain))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model pretrained loaded")
    
    #Set loss, optimizer and metric
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay = 1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    #Start a typical pytorch training/validation workflow
    max_epochs = opt.epochs
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    #writer = SummaryWriter()
    torch.backends.cudnn.benchmark = True

    for epoch in range(max_epochs):

        #####################
        ######TRAINING#######
        #####################
        print("-" * 10)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] epoch {epoch +1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = (batch_data['image'], batch_data['label'])
            if num_gpus > 0 :
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // opt.batch_size
            
            if (step -1)%10 == 0:
                print(
                    f"{step}/{epoch_len}, "
                    f"train_loss: {loss.item():.4f}")
            #writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        #####################
        #####VALIDATION######
        #####################
        if (epoch + 1)% val_interval == 0:
            model.eval()
            with torch.no_grad():

                #IN CASE OF saving code below
            
                for val_data in val_loader:
                    val_inputs, val_labels = (val_data['image'], val_data['label'])
                    if opt.gpus != '-1':
                        val_inputs = val_inputs.cuda()
                        val_labels = val_labels.cuda()
                    roi_size =  (im_size ,im_size ,im_size)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric= dice_metric.aggregate().item()
                dice_metric.reset()

                metric_values.append(metric)
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, save_name + ".pth"
                    ))
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New best metric model was saved ")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                    f" current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

                ###PLOT LOSS DICE RESULT
                loss_dice_plot = {
                            'epoch_loss_values':epoch_loss_values,
                            'val_interval':val_interval,
                            'metric_values': metric_values,
                            }

                plot(save_name + '.jpg', loss_dice_plot)

    ###PLOT LOSS DICE RESULT
    loss_dice_plot = {
                'epoch_loss_values':epoch_loss_values,
                'val_interval':val_interval,
                'metric_values': metric_values,
                }

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ROI training ...")
    new_train, new_val = [], []
    train(new_train, new_val)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training finished")
if __name__ == "__main__":
    main()