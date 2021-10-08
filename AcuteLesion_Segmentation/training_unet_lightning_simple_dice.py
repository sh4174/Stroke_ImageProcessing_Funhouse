import os
import shutil
import tempfile
import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    RandShiftIntensityd,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandAdjustContrastd,
    EnsureType,
    EnsureTyped
)

from monai.networks.layers import Norm

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
    pad_list_data_collate,
)

import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(0)
torch.backends.cudnn.benchmark = True
print_config()


# Output Directory
model_dir = 'models/monai_unet_lightning_crop192_dice/'
os.makedirs( model_dir, exist_ok=True )

# Set up Lightning Module
class UNet_Module(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()

        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose( [ EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=2) ] )
        self.post_label = Compose( [ EnsureType(), AsDiscrete(to_onehot=True, num_classes=2) ] )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 2000
        self.check_val = 50
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW( self._model.parameters(), lr=1e-4, weight_decay=1e-5 )
        optimizer = torch.optim.Adam( self._model.parameters(), lr=1e-4 )
        return optimizer


    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].cuda(), batch["label"].cuda())
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (192, 192, 16)
        sw_batch_size = 8
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}


    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}


# Main
# initialise the LightningModule
unet_module = UNet_Module()


# Set up training data 
img_dir = '/data/vision/polina/users/razvan/sungmin/data/Need_IRB_Approval/LesionSegmentation/Training_Resized/DWI/'
seg_dir = '/data/vision/polina/users/razvan/sungmin/data/Need_IRB_Approval/LesionSegmentation/Training_Resized/Lesion/'

images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
segs = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))

print( "========================================" )
print( "len images" )
print( len( images ) )
print( "========================================" )

train_files = [
    {"image": img, "label": seg} for img, seg in zip(images[100:], segs[100:])
]
print( "========================================" )
print( "len train files" )
print( len( train_files ) )
print( "========================================" )

val_files = [
    {"image": img, "label": seg} for img, seg in zip(images[:100], segs[:100])
]

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        RandAdjustContrastd( keys=["image"], prob=0.2 ),
        ScaleIntensityd(
            keys=["image"],
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.10,
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(192, 192, 16),
            random_size=False,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=8,
)


train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=list_data_collate,
)

# Validation
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(
            keys=["image"],
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=4,
)
val_loader = torch.utils.data.DataLoader( val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True )


# set up checkpoints
checkpoint_callback = ModelCheckpoint(dirpath=model_dir, filename="best_metric_model")

####################################
# Training
max_epochs = 2000
check_val = 50

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    gpus=[0],
    max_epochs=max_epochs,
    # max_steps=100,
    # limit_train_batches=100,
    check_val_every_n_epoch=check_val,
    callbacks=[checkpoint_callback],
    default_root_dir=model_dir,
    log_every_n_steps=1,
)

# train
trainer.fit(unet_module, train_loader, val_loader)

eval_num = 1
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(unet_module.epoch_loss_values))]
y = unet_module.epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(unet_module.metric_values))]
y = unet_module.metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.savefig( os.path.join(model_dir, "train_loss_val_metric.png") )

# Prediction / Validation
case_num = 4
# unet_module.load_from_checkpoint(os.path.join(model_dir, "unet_model_epoch=0.ckpt"))
unet_module.eval()
unet_module.to(device)

with torch.no_grad():
    img_name = os.path.split(
        val_ds[case_num]["image_meta_dict"]["filename_or_obj"]
    )[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).to(device)
    val_labels = torch.unsqueeze(label, 1).to(device)
    val_outputs = sliding_window_inference(
        val_inputs, (192, 192, 16), 8, unet_module, overlap=0.8
    )
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 16], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title(f"label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 16])
    plt.subplot(1, 3, 3)
    plt.title(f"output")
    plt.imshow(
        torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 16]
    )
    plt.savefig( os.path.join(model_dir, "net.png") )


with torch.no_grad():
    img_name = os.path.split(
        val_ds[case_num]["image_meta_dict"]["filename_or_obj"]
    )[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).to(device)
    val_labels = torch.unsqueeze(label, 1).to(device)
    val_outputs = sliding_window_inference(
        val_inputs, (192, 192, 16), 8, unet_module.forward, overlap=0.8
    )
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 16], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title(f"label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 16])
    plt.subplot(1, 3, 3)
    plt.title(f"output")
    plt.imshow(
        torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 16]
    )
    plt.savefig( os.path.join(model_dir, "net_forward.png") )


