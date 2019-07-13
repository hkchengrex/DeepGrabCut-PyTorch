import cv2
import numpy as np
import os
import sys
import torch

from torch.nn.functional import upsample
from dataloaders import utils
import networks.deeplab_resnet as resnet

from PIL import Image

img_dir = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)

gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
print("Initializing weights from: {}".format(os.path.join('models/', 'deepgc_pascal_epoch-99.pth')))
state_dict_checkpoint = torch.load(os.path.join('models/', 'deepgc_pascal_epoch-99.pth'),
                                   map_location=lambda storage, loc: storage)

net.load_state_dict(state_dict_checkpoint)
net.eval()
net.to(device)

files = os.listdir(img_dir)
imgs = [im for im in files if '_gt.png' not in im]

for im_name in imgs:

    im_path = os.path.join(img_dir, im_name)
    im = np.array(Image.open(im_path))
    img_shape = (450, 450)
    image = utils.fixed_resize(im, img_shape).astype(np.uint8)

    gt = Image.open(im_path.replace('.png', '_gt.png')).convert('L')
    gt = np.array(gt)

    # image = np.array(im)
    gt_bb = utils.get_bbox(gt)

    # output = cv2.resize(output, img_shape)

    # dismap = utils.compute_dismap(gt, gt_bb)
    dismap, tmp = utils.distance_map(gt, v=0)

    dismap[dismap > 255] = 255
    dismap[dismap < 0] = 0
    dismap = dismap

    dismap = utils.fixed_resize(dismap, (450, 450)).astype(np.uint8)

    dismap = np.expand_dims(dismap, axis=-1)

    image = image[:, :, ::-1] # change to rgb
    merge_input = np.concatenate((image, dismap), axis=2).astype(np.float32)
    inputs = torch.from_numpy(merge_input.transpose((2, 0, 1))[np.newaxis, ...])

    # Run a forward pass
    inputs = inputs.to(device)
    outputs = net.forward(inputs)
    outputs = upsample(outputs, size=(450, 450), mode='bilinear', align_corners=True)
    outputs = outputs.to(torch.device('cpu'))

    prediction = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
    prediction = 1 / (1 + np.exp(-prediction))
    prediction = np.squeeze(prediction)
    # prediction[prediction>thres] = 255
    # prediction[prediction<=thres] = 0

    cv2.imwrite(os.path.join(out_dir, im_name.replace('.png', '_mask.png')), prediction * 255)
    # cv2.imwrite(os.path.join(out_dir, im_name.replace('.jpg', '_dismap.png')), dismap)
    # cv2.imwrite(os.path.join(out_dir, im_name.replace('.jpg', '_tmp.png')), tmp * 255)
    # cv2.imwrite(os.path.join(out_dir, im_name.replace('.jpg', '_gt.png')), gt)

    # prediction = np.expand_dims(prediction, axis=-1).astype(np.uint8)
    # image = image[:, :, ::-1] # change to bgr
    # display_mask = np.concatenate([np.zeros_like(prediction), np.zeros_like(prediction), prediction], axis=-1)
    # image = cv2.addWeighted(image, 0.9, display_mask, 0.5, 0.1)