import os, sys
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
import h5py
import pathlib

from dataloader.semantickitti import SemanticKITTI

def load_dist(lidar, min_bound, max_bound, num_bins):
    bin_length = max_bound // num_bins
    assert(num_bins * bin_length == max_bound)
    dist = np.sqrt(lidar[:,0] ** 2 + lidar[:,1] ** 2)
    dist = dist.clip(min=min_bound, max=max_bound)
    dist = dist // bin_length
    dist = dist.clip(max=num_bins-1)
    return dist

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/crb.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    parser.add_argument('--save_dir', default='output')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    ds = SemanticKITTI(split='train', config=config['dataset'])
    num_classes = len(config['dataset']['labels'])
    p_value = config['p_value']
    k_count = config['k_count']
    hf = h5py.File(os.path.join(args.save_dir, 'training_results.h5'), 'r')

    # Initialize variables
    k_value = np.ones((k_count, num_classes))
    M = {}
    for j in range(k_count):
        M[j] = {}
        for i in range(num_classes):
            M[j][i] = {
                'conf': np.empty((len(ds)*104126,1)),
                'count': 0
            }

    # Get global lists of confidence, prediction and range
    print('Determining global threshold k^(c,r)...')
    for i in tqdm(range(len(ds))):
        label_path = ds.label_paths[i]
        pred = hf[os.path.join(label_path, 'pred')][()]
        conf = hf[os.path.join(label_path, 'conf')][()]
        lidar = ds.get_lidar(i)
        dist = load_dist(lidar, config['min_bound'], config['max_bound'], k_count)

        for k in range(k_count):
            bin_mask = dist == k
            if bin_mask.sum() == 0:
                continue
            for j in range(num_classes):
                mask = pred[bin_mask] == j
                if mask.sum() == 0:
                    continue
                bin_conf = conf[bin_mask][mask]
                start = M[k][j]['count']
                count = bin_conf.shape[0]
                M[k][j]['conf'][start:start+count] = bin_conf
                M[k][j]['count'] += count

    # Get CRB thresholds for class-annuli pairings
    for j in range(k_count):
        for i in range(num_classes):
            bin_count = M[j][i]['count']
            bin_conf = M[j][i]['conf'][:bin_count]
            sorted_conf = np.sort(bin_conf, 0)
            loc = int(np.round(bin_count * p_value))
            if loc == 0:
                continue
            index = max(-loc-1, -bin_count)
            k_value[j][i] = sorted_conf[index]

    # Generate pseudo-labels
    print(f'Generating pseudo-labels with beta={str(p_value*100)}%...')
    learning_map_inv = np.asarray(list(config['dataset']['learning_map_inv'].values()))
    for i in tqdm(range(len(ds))):
        label_path = ds.label_paths[i]
        pred = hf[os.path.join(label_path, 'pred')][()]
        conf = hf[os.path.join(label_path, 'conf')][()]
        lidar = ds.get_lidar(i)
        dist = load_dist(lidar, config['min_bound'], config['max_bound'], k_count)
        scribbles = ds.get_label(i)

        for l in range(k_count):
            for j in range(1, num_classes):
                mask = (dist == l) & (pred == j) & (conf > k_value[l][j]) & (scribbles == 0)
                scribbles[mask] = j

        # Save pseudo-labels
        true_label = learning_map_inv[scribbles].astype(np.uint32)
        crb_path = pathlib.Path(label_path.replace('scribbles', 'crb'))
        crb_path.parents[0].mkdir(parents=True, exist_ok=True)
        true_label.tofile(crb_path)
    hf.close()