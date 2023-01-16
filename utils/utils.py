import yaml
import torch
from torch import optim
from models.CDCNs import CDCN, CDCNpp
import cv2
import os
import logging

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg


def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    else:
        raise NotImplementedError

    return optimizer


def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    device = None
    if cfg['device'] == '':
        device = torch.device("cpu")
    elif cfg['device'] == '0':
        device = torch.device("cuda:0")
    elif cfg['device'] == '1':
        device = torch.device("cuda:1")
    else:
        raise NotImplementedError
    return device


def build_network(cfg):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    network = None

    if cfg['model']['base'] == 'CDCNpp':
        network = CDCNpp()
    elif cfg['model']['base'] == 'CDCN':
        network = CDCN()
    else:
        raise NotImplementedError

    return network

def read_image(image_path):
    """
    Read an image from input path
    params:
        - image_local_path (str): the path of image.
    return:
        - image: Required image.
    """

    # image_path = LOCAL_ROOT + image_path

    img = cv2.imread(image_path)
    # Get the shape of input image
    real_h,real_w,c = img.shape
    assert os.path.exists(image_path[:-4] + '_BB.txt'),'path not exists' + ' ' + image_path
    
    with open(image_path[:-4] + '_BB.txt','r') as f:
        material = f.readline()
        try:
            x,y,w,h,score = material.strip().split(' ')
        except:
            logging.info('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')   

        try:
            w = int(float(w))
            h = int(float(h))
            x = int(float(x))
            y = int(float(y))
            w = int(w*(real_w / 224))
            h = int(h*(real_h / 224))
            x = int(x*(real_w / 224))
            y = int(y*(real_h / 224))

            # Crop face based on its bounding box
            y1 = 0 if y < 0 else y
            x1 = 0 if x < 0 else x 
            y2 = real_h if y1 + h > real_h else y + h
            x2 = real_w if x1 + w > real_w else x + w
            img = img[y1:y2,x1:x2,:]

        except:
            logging.info('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img