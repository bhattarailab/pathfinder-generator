## Arguments from LRA repository:
### https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/pathfinder.py

import time
import sys
import numpy as np
import os
import snakes2

class Args:
    def __init__(self,
                 contour_path = './contour', batch_id=0, n_images = 200000,
                 window_size=[256,256], padding=22, antialias_scale = 4,
                 LABEL =1, seed_distance= 27, marker_radius = 3,
                 contour_length=15, distractor_length=5, num_distractor_snakes=6, snake_contrast_list=[1.], use_single_paddles=False,
                 max_target_contour_retrial = 4, max_distractor_contour_retrial = 4, max_paddle_retrial=2,
                 continuity = 1.4, paddle_length=5, paddle_thickness=1.5, paddle_margin_list=[4], paddle_contrast_list=[1.],
                 pause_display=False, save_images=True, save_metadata=True):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.padding = padding
        self.antialias_scale = antialias_scale

        self.LABEL = LABEL
        self.seed_distance = seed_distance
        self.marker_radius = marker_radius
        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.num_distractor_snakes = num_distractor_snakes
        self.snake_contrast_list = snake_contrast_list
        self.use_single_paddles = use_single_paddles

        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial

        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list # if multiple elements in a list, a number will be sampled in each IMAGE
        self.paddle_contrast_list = paddle_contrast_list # if multiple elements in a list, a number will be sampled in each PADDLE

        self.pause_display = pause_display
        self.save_images = save_images
        self.save_metadata = save_metadata

## Constraints
num_machines = int(sys.argv[1])
current_id = int(sys.argv[2])
total_images = int(sys.argv[3])
### sys.argv[4] contains task information (used below)
task = sys.argv[4]
seed = int(sys.argv[5])
assert seed >= 0 and seed <=10000, "Please Enter seed >= 0 and <=10000"

### Dataset Paths
dataset_root = './generation'
if len(sys.argv)==6:
    print('Using default path...')
elif len(sys.argv)==7:
    print('Using custom save path...')
    dataset_root = str(sys.argv[6])


### Dataset Configs

##### Long Range Arena (LRA) configurations are in:
##### https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/pathfinder.py

def get_pf64u_cl14_nogap_args():
    '''
    This function generates pathfinder 128 in complexity of path64.
    Path64 has paddle_margin = 1 whereas Path128 has  = [2, 3]
    '''

    args = Args()

    args.padding = 1
    args.paddle_margin_list = [1]
    args.seed_distance = 20
    args.window_size = [128,128]
    args.marker_radius = 3
    args.contour_length = 14
    args.paddle_thickness = 1.5
    args.antialias_scale = 2
    args.continuity = 1.8  # from 1.8 to 0.8, with steps of 66%
    args.distractor_length = args.contour_length / 3
    args.num_distractor_snakes = 22 / args.distractor_length
    args.snake_contrast_list = [0.9]
    return args

##### Interpolate Dataset Between Complexity of Pathfinder64 and PathfinderX

### This is base configuration
def get_pf64u_cl14_with_gap_args():
    '''
    This function generates pathfinder 128 in complexity of path64 but with 
    paddle_margin = 2 whereas No_Gap has  = 1
    '''
    
    args = get_pf64u_cl14_nogap_args()
    args.paddle_margin_list = [2, 2]
    
    return args

#### This is pathX configuration
def get_pf128_cl14_pathx_args():
    '''
    The configuration is taken from original source and modified.
    Num_distractor_snakes is taken from: https://github.com/google-research/long-range-arena/issues/38#issuecomment-947119529
    '''
    
    args = get_pf64u_cl14_nogap_args()
    
    args.paddle_margin_list = [2,3]
    args.distractor_length = args.contour_length / 3
    args.num_distractor_snakes = 35*2 / args.distractor_length
    return args

### This function merges the configurations of two configs
def get_merge_cl14_args(A, B, alpha=0.5):
    args = Args()
    a = 1-alpha
    b = alpha

    args.padding = int(np.round(a*A.padding + b*B.padding))
    args.antialias_scale = int(np.round(a*A.antialias_scale + b*B.antialias_scale))
    
    args.paddle_margin_list = [int(np.round(a*A.paddle_margin_list[0]+b*B.paddle_margin_list[0])),
                               int(np.round(a*A.paddle_margin_list[1]+b*B.paddle_margin_list[1]))]
    args.seed_distance = int(a*A.seed_distance + b*B.seed_distance)
    args.window_size = [128,128]
    args.marker_radius = a*A.marker_radius + b*B.marker_radius
    args.paddle_thickness = a*A.paddle_thickness + b*B.paddle_thickness
    args.paddle_thickness = float(int(np.round(args.paddle_thickness*args.antialias_scale)))/args.antialias_scale

    args.continuity = a*A.continuity + b*B.continuity
    
    args.contour_length = 14
    args.distractor_length = args.contour_length / 3

    args.num_distractor_snakes = a*A.num_distractor_snakes + b*B.num_distractor_snakes
    args.snake_contrast_list = [a*A.snake_contrast_list[0] + b*B.snake_contrast_list[0]]

    return args


args = Args()
if str(task).strip() == "nogap":
    args = get_pf64u_cl14_nogap_args()
    dataset_subpath = 'cl14_nogap'
else: 
    alpha = float(task)
    args = get_merge_cl14_args(get_pf64u_cl14_with_gap_args(), get_pf128_cl14_pathx_args(), alpha)
    dataset_subpath = 'cl14_alpha'+str(alpha)

args.batch_id = current_id
args.n_images = int(np.ceil(float(total_images)/num_machines))
args.seed = seed
args.segmentation_task = False
args.segmentation_task_double_circle = False

args.contour_path = os.path.join(dataset_root, dataset_subpath)
   
t = time.time()
snakes2.from_wrapper(args)

elapsed = time.time() - t
print('n_totl_imgs (per condition) : ', str(total_images))
print('ELAPSED TIME OVER ALL CONDITIONS : ', str(elapsed))
