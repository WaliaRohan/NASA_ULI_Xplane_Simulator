"""
modified from evaluate_pretrained_tiny_imagenet.py
"""

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

# make sure this is a system variable in your bashrc
# NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
NASA_ULI_ROOT_DIR = '/home/speedracer1702/Projects/automata_lab/NASA_ULI_Xplane_Simulator/'
DATA_DIR = NASA_ULI_ROOT_DIR + 'data/'

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

NNET_UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/simulation/'
sys.path.append(NNET_UTILS_DIR)
from nnet import *
from textfile_utils import *


def get_data(condition, train_test_split):

    print('evaluating: ', condition, train_test_split)

    # where original XPLANE images are stored 
    label_file = DATA_DIR + 'downsampled/' + condition + '/' + '_'.join([condition, train_test_split, 'downsampled']) + '.h5'

    f = h5py.File(label_file, "r")

    # get data
    x_train = f['X_train'][:] # Image
    y_train = f['y_train'][:] # Groundtruth (CTE, Heading)

    f.close()
    
    print(' ')
    print('X_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)
    print(' ')

    num_images_total = x_train.shape[0]

    # to get a subset of data
    start_index = np.random.randint(num_images_total - MAX_IMAGES)
    start_index = 0

    x_array_np = x_train[start_index: start_index + MAX_IMAGES]
    y_array_np = y_train[start_index: start_index + MAX_IMAGES]

    # to use all data
    x_array_np = x_train
    y_array_np = y_train

    return x_array_np, y_array_np

if __name__ == '__main__':

    prefix = 'evaluate_tiny_taxinet'
    train_test_split_list = ['train', 'validation', 'test']

    MAX_IMAGES = 10
    NROW = 5

    # Read in the pretrained tiny taxinet DNN
    filename = "../../models/TinyTaxiNet.nnet"
    network = NNet(filename)

    # condition: 'afternoon', 'morning', 'night', 'overcast'
    condition = 'morning'

    # create a temp dir to visualize a few images
    visualization_dir = SCRATCH_DIR + '/' + prefix + '/' + condition
    remove_and_create_dir(visualization_dir)

    prediction_error_list = [] # Predicted CTE - Ground Truth CTE
    closest_cte_list = [] # Ground truth CTE - tread as OOD

    # Get training data
    x_train, y_train = get_data(condition, 'train')

    # Get testing data
    x_array_np, y_array_np = get_data(condition, 'test')
    
    # evaluate the DNN for a single input
    for idx in range(x_array_np.shape[0]):
        pred = network.evaluate_network(x_array_np[idx].flatten())
        target = y_array_np[idx]
        target_pred = target[0:2]
        
        # Find predicted CTE - gt CTE
        # prediction_error = np.linalg.norm(target_pred - pred)
        prediction_error = target[0] - pred[0]
        gt_cte = target[0]

        # Append error and target's first value to different lists
        prediction_error_list.append(prediction_error)
        closest_cte_list.append(gt_cte)

    # Plot error on y axis and target on x axis

    # Convert to numpy arrays for plotting
    prediction_error_array = np.array(prediction_error_list)
    gt_cte_array = np.array(closest_cte_list)

    prediction_error_array = np.abs(prediction_error_array)
    gt_cte_array = np.abs(gt_cte_array)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(gt_cte_array, prediction_error_array, alpha=0.5, s=10)
    plt.xlabel("Ground Truth CTE")
    plt.ylabel("Prediction Error")
    plt.title(f"Prediction Error vs Ground Truth CTE [{condition}]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()