import numpy as np 
import pandas as pd 

import math
import random
import os 
import cv2
import timm

from tqdm import tqdm

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F 

import gc
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

class CFG:

	img_size = 512
	batch_size = 12
	seed = 2020

	classes = 11014

	scale = 30
	margin = 0.5

def read_dataset():
	df = pd.read_csv('')

if __name__ == '__main__':