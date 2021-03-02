import lmdb
import sys
import six
import re
import math

from PIL import Image 

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LmdbDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            self.num_samples = num_samples

        # Data Filtering: off
            self.filtered_index_list = [index + 1 for index in range(self.num_samples)]
            
        # Data Filtering: on
            # self.filtered_index_list = []
            # for index in range(self.num_samples):
            #     index += 1 # lmdb starts with 1
            #     label_key = 'label-%09d'.encode() % index
            #     label = txn.get(label_key).decode('utf-8')

            #     out_of_char = '^[ㄱ-ㅎ|가-힣|a-z|A-Z|0-9|\s]+$'
            #     if re.search(out_of_char, label):
            #         print('out: ', label)
            #         continue

            #     self.filtered_index_list.append(index)

            # self.num_samples = len(self.filtered_index_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                # if self.opt.rgb:
                #     img = Image.open(buf).convert('RGB')  # for color image
                # else:
                img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                # if self.opt.rgb:
                #     img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                # else:
                #     img = Image.new('L', (self.imgW, self.imgH))
                img = Image.new('L', (32, 100))
                label = '[dummy_label]'

        return (img, label)


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        # if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
        #     resized_max_w = self.imgW
        #     input_channel = 3 if images[0].mode == 'RGB' else 1
        #     transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        #     resized_images = []
        #     for image in images:
        #         w, h = image.size
        #         ratio = w / float(h)
        #         if math.ceil(self.imgH * ratio) > self.imgW:
        #             resized_w = self.imgW
        #         else:
        #             resized_w = math.ceil(self.imgH * ratio)

        #         resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
        #         resized_images.append(transform(resized_image))

        #     image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        # else:
        transform = ResizeNormalize((self.imgW, self.imgH))
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

# class NormalizePAD(object):

#     def __init__(self, max_size, PAD_type='right'):
#         self.toTensor = transforms.ToTensor()
#         self.max_size = max_size
#         self.max_width_half = math.floor(max_size[2] / 2)
#         self.PAD_type = PAD_type

#     def __call__(self, img):
#         img = self.toTensor(img)
#         img.sub_(0.5).div_(0.5)
#         c, h, w = img.size()
#         Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
#         Pad_img[:, :, :w] = img  # right pad
#         if self.max_size[2] != w:  # add border Pad
#             Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

#         return Pad_img