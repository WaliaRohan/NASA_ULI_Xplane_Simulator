import os
import torch

from model_taxinet import TaxiNetDNN
from taxinet_dataloader import *

from tqdm import tqdm

from textfile_utils import *

# Config
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
DATA_DIR = os.environ['NASA_DATA_DIR']
SCRATCH_DIR = NASA_ULI_ROOT_DIR + 'scratch'
dataset_type = 'large_images' # large_images or downsampled
condition = 'afternoon' # 'afternoon', 'morning', 'night', 'overcast'


def predict(model, dataset, dataloader, device):
    """Returns a dataframe of predictions and targets, one row per image in dataset.image_list"""

    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            outputs = model(inputs.to(device))
            preds.append(outputs.cpu().numpy())
            targets.append(labels.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # dataloader is not shuffled, so rows are in the same order as dataset.image_list
    return pandas.DataFrame({
        'image_filename': dataset.image_list,
        'true_cte_norm': targets[:, 0],
        'true_dtp_norm': targets[:, 1],
        'pred_cte_norm': preds[:, 0],
        'pred_dtp_norm': preds[:, 1],
    })


if __name__ == '__main__':

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('found device: ', device)

    results_dir = remove_and_create_dir(SCRATCH_DIR + '/test_DNN_taxinet_occlusion')

    # Dataloader params
    dataloader_params = {'batch_size': 128,
                        'shuffle': False,
                        'num_workers': 12,
                        'drop_last': False}

    test_dir = os.path.join(DATA_DIR, dataset_type, condition, f"{condition}_test") # Source image files
    occ_dir = os.path.join(DATA_DIR, dataset_type, f"{condition}_occ", f"{condition}_occ_test") # Save dir for occluded images

    # Load Model
    model_dir = NASA_ULI_ROOT_DIR + '/pretrained_DNN/' 
    model = TaxiNetDNN()

    ## load the pre-trained model
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_dir + '/best_model.pt'))

    model = model.to(device)
    model.eval()
    
    # Load datasets
    occ_images = sorted([x for x in os.listdir(occ_dir) if x.endswith('.png')])

    test_dataset = TaxiNetDataset(test_dir)
    test_dataset.image_list = occ_images

    # the occluded dir has no labels.csv, so build the dataset from the clean
    # test dir (for its labels) and point it at the occluded images instead
    occ_dataset = TaxiNetDataset(test_dir)
    occ_dataset.data_dir = occ_dir
    occ_dataset.image_list = occ_images

    # Predict
    test_df = predict(model, test_dataset, DataLoader(test_dataset, **dataloader_params), device)
    occ_df = predict(model, occ_dataset, DataLoader(occ_dataset, **dataloader_params), device)

    occ_df = occ_df[['image_filename', 'pred_cte_norm', 'pred_dtp_norm']].rename(
    columns={'pred_cte_norm': 'pred_cte_norm_occ', 'pred_dtp_norm': 'pred_dtp_norm_occ'})
    results_df = test_df.merge(occ_df, on='image_filename', how='inner')

    out_csv = results_dir + '/' + condition + '_' + dataset_type + '_occlusion_predictions.csv'
    results_df.to_csv(out_csv, index=False)

    print('Occluded images: ', len(results_df))
    print('Saved predictions to: ', out_csv)
