# Permalink to original code: https://github.com/NVIDIA-Omniverse/synthetic-data-examples/blob/78622588948e055e27aa8b0ef8494a73855bceeb/end-to-end-workflows/object_detection_fruit/training/code/train.py
# Changes:
#   - Removed the training logic, and added code to run the model on a dataset
#   - Changed some args parser options

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
import torchvision
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes, save_image

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

        self.static_labels = [
            "apple",
            "avocado",
            "lime"
        ]
        self.num_classes = len(self.static_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "png", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        print(f'{idx}: {img_path}')

        if self.transforms is not None:
            img = self.transforms(img)
        target = {}
        return img, target

    def __len__(self):
        return len(self.imgs)


"""
Parses command line options. Requires input data directory, output torch file, and number epochs used to train.
"""
def parse_input():
    usage = "usage: eval.py [options] arg1 arg2 arg3"
    parser = OptionParser(usage)
    parser.add_option(
        "-d",
        "--data_dir",
        dest="data_dir",
        help="Directory location for Omniverse synthetic data.",
    )
    parser.add_option(
        "-o",
        "--output_folder",
        dest="output_folder",
        help="Save annotated images to this location",
    )
    parser.add_option(
        "-m",
        "--model",
        dest="model_file",
        help="Trained model file (.pth)",
    )
    (options, args) = parser.parse_args()
    return options, args


def get_transform():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(model_file, num_classes, device):
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    return model


def decode_output(output, labels_list, threshold=0.05):
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([labels_list[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    idxs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), threshold)
    bbs, confs, labels = [tensor[idxs] for tensor in [bbs, confs, labels]]
    if len(idxs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

# Constants
NMS_THRESHOLD = 0.05

def main():
    options, args = parse_input()
    dataset = FruitDataset(options.data_dir, get_transform())

    validloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is available.")
    else:
        device = torch.device("cpu")
        print("Cuda is not available, training with cpu.")

    # Load model from file
    model = load_model(options.model_file, dataset.num_classes, device)
    model.eval()
    for i, (imgs, annotations) in enumerate(validloader):
        imgs = list(img.to(device) for img in imgs)
        outputs = model(imgs)
        for j, output in enumerate(outputs):
            byteim = torch.mul(imgs[j], 255).byte()
            bbs, confs, labels = decode_output(output, dataset.static_labels, NMS_THRESHOLD)
            bbstensor = torch.tensor(bbs)
            anotated_im = draw_bounding_boxes(byteim, bbstensor, labels)
            anotated_im = torch.div(anotated_im, 255.).float()
            save_image(anotated_im, f'{options.output_folder}/output_{i}_{j}.jpg')


if __name__ == "__main__":
    main()
