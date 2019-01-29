import os
import sys
from PIL import Image
import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class CIFAR10(dset.CIFAR10):

  def __init__(self, root, train=True,
           transform=None, target_transform=None,
           download=False,validate_seed=0,
           val_split=0.1):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    self.val_split = val_split

    if download:
      self.download()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')

    # now load the picked numpy arrays
    
    
    self.data = []
    self.labels= []
    for fentry in self.train_list:
      f = fentry[0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data.append(entry['data'])
      if 'labels' in entry:
        self.labels += entry['labels']
      else:
        self.labels += entry['fine_labels']
      fo.close()
        
    self.data = np.concatenate(self.data)
    # Randomly select indices for validation
    if self.val_split > 0:
      label_indices = [[] for _ in range(max(self.labels)+1)]
      for i,l in enumerate(self.labels):
        label_indices[l] += [i]  
      label_indices = np.asarray(label_indices)
      
      # randomly grab 500 elements of each class
      np.random.seed(validate_seed)
      
      self.val_indices = []           
      
       
      
      for l_i in label_indices:
        self.val_indices += list(l_i[np.random.choice(len(l_i), int(len(self.data) * val_split) // (max(self.labels) + 1) ,replace=False)])

        
    if self.train=='validate':    
      self.data = self.data[self.val_indices]
      self.labels = list(np.asarray(self.labels)[self.val_indices])
      
      self.data = self.data.reshape((int(50e3 * self.val_split), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
    elif self.train:
      print(np.shape(self.data))
      if self.val_split > 0:
        self.data = np.delete(self.data,self.val_indices,axis=0)
        self.labels = list(np.delete(np.asarray(self.labels),self.val_indices,axis=0))
          
      self.data = self.data.reshape((int(50e3 * (1.-self.val_split)), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()
      self.data = self.data.reshape((10000, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
      
  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target
      
  def __len__(self):
      return len(self.data)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    
    # Only need to subclass STL10 to make __init__ args the same as CIFAR
class STL10(dset.STL10):
        def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
            super(STL10, self).__init__(root, split='train' if train else 'test',
                 transform=transform, target_transform=target_transform, download=download)
            self.fold = fold
            fold_indices = np.int64(np.loadtxt('STL/stl10_binary/fold_indices.txt'))
            
            if train:
                if fold != 10: # there are 10 folds so if fold is 10 we presume we're using all
                    print('using fold #%i...'%fold)
                    self.data = np.asarray(self.data)[fold_indices[fold]]
                    self.labels = np.asarray(self.labels)[fold_indices[fold]]
                self.train_data = self.data
                self.train_labels = self.labels
            else:
                self.test_data = self.data
                self.test_labels = self.labels
                
                
                
# ImageNet32x32 dataset.
class I1000(data.Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
         
        if self.train:
            self.train_data = np.load(self.root + '/imagenet32_train.npz')['data']
            self.train_labels = np.load(self.root + '/imagenet32_train.npz')['labels']
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        else:
            self.test_data = np.load(self.root + '/imagenet32_val.npz')['data']
            self.test_labels = np.load(self.root + '/imagenet32_val.npz')['labels']
            self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
            
class MN40(data.Dataset):
    """`Modelnet-40 <modelnet.cs.princeton.edu>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``modelnet40`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        num_rotations: how many rotations of each example to use.
    """


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 num_rotations=12):
        self.root = root
        self.transform = self.jitter_chunk
        self.train = train  # training set or test set
        self.num_rotations = num_rotations # how many rotations to train on


        # now load the picked numpy arrays
        if self.train:
            self.train_data = np.load(self.root + '/modelnet40_rot24_train.npz')['data'][:9840]
            self.train_labels = np.load(self.root + '/modelnet40_rot24_train.npz')['labels'][:9840]
        else:
            self.test_data = np.load(self.root + '/modelnet40_rot24_test.npz')['data']
            self.test_labels = np.load(self.root + '/modelnet40_rot24_test.npz')['labels']


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Assume for now we're randomly sampling rotations.
        if self.train:
            img, target = self.train_data[index,[np.random.choice(range(0, 24, 24//self.num_rotations))]], self.train_labels[index]
            if self.transform is not None:    
                img = self.transform(img)
        else:
            # Grab lots of rotated examples of instance but note that we will be averaging across rotations so no need to duplicate labels.
            # img = np.asarray([individual_rotation for all_rotations in self.test_data[index] 
                                                          # for individual_rotation in all_rotations[range(0, 24, 24//self.num_rotations)]])
            img = np.asarray([individual_rotation for individual_rotation in self.test_data[index,range(0, 24, 24//self.num_rotations)]])
            target = self.test_labels[index]

        
            
        img = torch.from_numpy(np.float32(img) * 6 - 1)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    # Chunk Jiterrer for Modelnet data augmentation
    def jitter_chunk(self, src):

        dst = src.copy()
        if np.random.binomial(1, .2):
            dst[ :, ::-1, :, :] = dst
        if np.random.binomial(1, .2):
            dst[ :, :, ::-1, :] = dst
        max_ij = 2
        max_k = 2
        shift_ijk = [np.random.randint(-max_ij, max_ij),
                     np.random.randint(-max_ij, max_ij),
                     np.random.randint(-max_k, max_k)]
        for axis, shift in enumerate(shift_ijk):
            if shift != 0:
                # beware wraparound
                dst = np.roll(dst, shift, axis+1)
        return dst
class MN10(MN40):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 num_rotations=12):
        self.root = root
        self.transform = jitter_chunk
        self.train = train  # training set or test set
        self.num_rotations = num_rotations # how many rotations to train on


        # now load the picked numpy arrays
        if self.train:
            self.train_data = np.load(self.root + '/modelnet10_rot24_train.npz')['data'][:3990]
            self.train_labels = np.load(self.root + '/modelnet10_rot24_train.npz')['labels'][:3990]
        else:
            self.test_data = np.load(self.root + '/modelnet10_rot24_test.npz')['data']
            self.test_labels = np.load(self.root + '/modelnet10_rot24_test.npz')['labels']
            
""" ILSVRC """
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
  images = []
  dir = os.path.expanduser(dir)
  for target in tqdm(sorted(os.listdir(dir))):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)

  return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)


class ImageFolder(data.Dataset):
  """A generic data loader where the images are arranged in this way: ::

      root/dog/xxx.png
      root/dog/xxy.png
      root/dog/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      classes (list): List of the class names.
      class_to_idx (dict): Dict with items (class_name, class_index).
      imgs (list): List of (image path, class_index) tuples
  """

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False, **kwargs):
    classes, class_to_idx = find_classes(root)
    if os.path.exists('imagenet_imgs.npz'):
      print('Loading pre-saved ImageNet Index file...')
      imgs = np.load('imagenet_imgs.npz')['imgs']
    else:
      print('Generating ImageNet Index file...')
      imgs = make_dataset(root, class_to_idx)
      np.savez_compressed('imagenet_imgs.npz', **{'imgs' : imgs})
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem
    
    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = self.transform(imgs[index][0]), imgs[index][1]
        self.data.append(self.loader(path))
        self.labels.append(target)
          

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    if self.load_in_mem:
        img = self.data[index]
        target = self.labels[index]
    else:
      path, target = self.imgs[index]
      img = self.loader(str(path))
      if self.transform is not None:
        img = self.transform(img)
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    # print(img.size(), target)
    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
