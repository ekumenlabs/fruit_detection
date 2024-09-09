# Permalink to original code: https://github.com/NVIDIA-Omniverse/synthetic-data-examples/blob/78622588948e055e27aa8b0ef8494a73855bceeb/end-to-end-workflows/object_detection_fruit/training/code/train.py
# Changes:
#   - Modified the amount of labels
#   - Made static_labels part of the dataset class
#   - Removed the epochs option from the args parser and moved it to a constant
#   - Moved some training values to constants
#   - Added some logging

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import json
import numpy as np
from optparse import OptionParser
import os
from PIL import Image
import shutil
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        list_ = os.listdir(root)
        for file_ in list_:
            name, ext = os.path.splitext(file_)
            ext = ext[1:]
            if ext == "":
                continue

            if os.path.exists(root + "/" + ext):
                shutil.move(root + "/" + file_, root + "/" + ext + "/" + file_)

            else:
                os.makedirs(root + "/" + ext)
                shutil.move(root + "/" + file_, root + "/" + ext + "/" + file_)

        self.imgs = list(sorted(os.listdir(os.path.join(root, "png"))))
        self.label = list(sorted(os.listdir(os.path.join(root, "json"))))
        self.box = list(sorted(os.listdir(os.path.join(root, "npy"))))
        
        self.static_labels = {
            "apple": 0,
            "avocado": 1,
            "lime": 2,
        }
        self.num_classes = len(self.static_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "png", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.root, "json", self.label[idx])

        with open(label_path, "r") as json_data:
            json_labels = json.load(json_data)

        box_path = os.path.join(self.root, "npy", self.box[idx])
        dat = np.load(str(box_path))

        boxes = []
        labels = []
        for i in dat:
            obj_val = i[0]
            xmin = torch.as_tensor(np.min(i[1]), dtype=torch.float32)
            xmax = torch.as_tensor(np.max(i[3]), dtype=torch.float32)
            ymin = torch.as_tensor(np.min(i[2]), dtype=torch.float32)
            ymax = torch.as_tensor(np.max(i[4]), dtype=torch.float32)
            if (ymax > ymin) & (xmax > xmin):
                boxes.append([xmin, ymin, xmax, ymax])
                area = (xmax - xmin) * (ymax - ymin)
            labels += [json_labels.get(str(obj_val)).get("class")]

        label_dict = {}
        labels_out = []

        for i in range(len(labels)):
            label_dict[i] = labels[i]

        for i in label_dict:
            fruit = label_dict[i]
            final_fruit_label = self.static_labels[fruit]
            labels_out += [final_fruit_label]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels_out, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = area

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


"""
Parses command line options. Requires input data directory, output torch file, and number epochs used to train.
"""
def parse_input():
    usage = "usage: train.py [options] arg1 arg2 "
    parser = OptionParser(usage)
    parser.add_option(
        "-d",
        "--data_dir",
        dest="data_dir",
        help="Directory location for Omniverse synthetic data.",
    )
    parser.add_option(
        "-o",
        "--output_file",
        dest="output_file",
        help="Save torch model to this file and location (file ending in .pth)",
    )
    (options, args) = parser.parse_args()
    return options, args


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Constants
NUM_EPOCHS = 5
TRAINING_PARTITION_RATIO = 0.7
OPTIMIZER_LR = 0.001
OPTIMIZER_MOMENTUM = 0.9
OPTIMIZER_WEIGHT_DECAY = 0.0005

def main():
    writer = SummaryWriter()
    options, args = parse_input()
    dataset = FruitDataset(options.data_dir, get_transform(train=True))
    train_size = int(len(dataset) * TRAINING_PARTITION_RATIO)
    unused_size = len(dataset) - train_size

    train, unused = torch.utils.data.random_split(
        dataset, [train_size, unused_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is available.")
    else:
        device = torch.device("cpu")
        print("Cuda is not available, training with cpu.")

    model = create_model(dataset.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=OPTIMIZER_LR, momentum=OPTIMIZER_MOMENTUM, weight_decay=OPTIMIZER_WEIGHT_DECAY)
    len_dataloader = len(train_loader)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        i = 0
        for imgs, annotations in train_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            writer.add_scalar("Loss/train", losses, epoch)

            losses.backward()
            optimizer.step()

            print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")

    writer.close()
    torch.save(model.state_dict(), options.output_file)
    print(f'Training ended. Model state dictionary was saved to file {options.output_file}.')


if __name__ == "__main__":
    main()
