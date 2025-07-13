from absl import app
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
import logging
import traceback
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.cluster import KMeans
import math
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.semi_supervised import LabelSpreading
from scipy.special import rel_entr  # Für KL-Divergenz

class ImageDataset(Dataset):
    def __init__(self, paths, transform, root):
        self.paths = paths
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(self.root + path).convert('RGB')
        return self.transform(image)


class SemanticInitSpreading(AlgorithmSkelton):
    def __init__(self):
        name = "semantic_init_spreading"
        AlgorithmSkelton.__init__(self, name)

        # ##### ResNet50
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        # Alles bis zum avgpool nehmen und flatten:
        self.model = torch.nn.Sequential(
            *(list(model.children())[:-1])  # Bis einschließlich avgpool
        )
        self.transform = weights.transforms()

    def extractFeatures(self, unlabeled_paths):

        dataset = ImageDataset(unlabeled_paths, self.transform, "/workspace/Data-Centric-Image-Classification/raw_datasets/")
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        features = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                output = self.model(batch)
                output = output.view(output.size(0), -1)  # Robust flatten
                features.append(output.cpu().numpy())
                if (i+1) % 5 == 0:
                    print(f'{i+1}/{len(loader)}')

        features = np.concatenate(features, axis=0)
        return features

    def run(self, ds, oracle, dataset_info, v_fold, num_annos, percentage_labeled):
        try:

            
    # 1. Setup
            unlabeled_paths, _ = ds.get_training_subsets('unlabeled')
            all_paths, _ = ds.get_training_subsets('all')
            n_unlabeled = len(unlabeled_paths)
            n_all = len(all_paths)
            n_init = int(1*n_unlabeled)
            n_active = 1-n_init

            nc = len(dataset_info.classes)  # Number of classes.
            p = 2                           # How often to label one image.
            k_clusters = nc*4               # Number of clusters for kmeans.
            k_cluster_call = n_init // (k_clusters*p)

            print(f'total: {k_cluster_call*k_cluster_call*p}')
            

            alpha = 0.2                     #Small: Labels tend to stay the same
            gamma = 0                       #Small: Global...

            # p_dict = {2: 2, 6: 2, 8: 2, 3:2}  # This could be used to select the right number of Labels
                                                # per image, for the respective model. My quick 
                                                # Hyperparameter search found this to be the best.

            print(f'n_all: {n_all}, n_unlabeled: {n_unlabeled}, k_cluster_call: {k_cluster_call}')

            # 2. Initialisation
            budget = n_unlabeled                        # Counter.
            labeled_paths = set()             # als Set für schnelleres Suchen!

            # 2.1 Extracting the features.
            features = self.extractFeatures(unlabeled_paths=unlabeled_paths)

            # 2.3 KMeans Clustering
            kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(features)
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            # 2.4 Get Center Images per Cluster
            top_n_idx = set()
            for cluster_id in range(k_clusters):
                indices = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
                if not indices:
                    continue
                # Distances to centers.
                cluster_feats = features[indices]
                center = cluster_centers[cluster_id]
                dists = np.linalg.norm(cluster_feats - center, axis=1)
                sorted_indices = np.argsort(dists)
                num = len(sorted_indices)
                if num < k_cluster_call:
                    chosen = sorted_indices  # All, if not enough
                else:
                    mid = num // 2
                    half = k_cluster_call // 2
                    if k_cluster_call % 2 == 0:
                        chosen = sorted_indices[mid-half:mid+half]
                    else:
                        chosen = sorted_indices[mid-half:mid+half+1]

                for idx_in_cluster in chosen:
                    global_idx = indices[idx_in_cluster]
                    path = unlabeled_paths[global_idx]
                    if path in labeled_paths:
                        continue
                    top_n_idx.add(global_idx)
                    labeled_paths.add(path)
                    budget-=p

            print(f'budget after cluster_init: {len(top_n_idx)*p/n_unlabeled:.3f}')

            # Rest of Budget.

            for i in range(len(unlabeled_paths)):
                if budget > 0:
                    if p > budget:
                        break
                    path = unlabeled_paths[i]
                    if path in labeled_paths:
                        continue
                    top_n_idx.add(i)
                    labeled_paths.add(path)
                    budget-=p


            # 2.5 Asking the Oracle
            labeled_indices = []
            labeled_labels = []
            oracle_count = 0
            labeled = 0
            for i in top_n_idx:
                path = unlabeled_paths[i]
                org_split = ds.get(path, 'original_split')
                oracle_label = [float(x) for x in oracle.get_soft_gt(path, p)]
                oracle_count += p
                ds.update_image(path, org_split, oracle_label)
                labeled += 1
                labeled_indices.append(i)
                labeled_labels.append(oracle_label)
            print(f'oracle_count: {oracle_count/n_unlabeled:.3f}')
    

    # 3. Method Here we skip the method part.


    # 4. Label Spreading
            pseudos=0
            int_labels = np.full(n_unlabeled, -1)
            for idx, lbl in zip(labeled_indices, labeled_labels):
                int_labels[idx] = int(np.argmax(lbl))


            label_spread = LabelSpreading(kernel='rbf', alpha=alpha, max_iter=40, gamma=gamma)
            label_spread.fit(features, int_labels)
            probas = label_spread.label_distributions_

            for i, path in enumerate(unlabeled_paths):
                if i not in labeled_indices:
                    org_split = ds.get(path, 'original_split')
                    pseudo_label = list(map(float, probas[i]))
                    # print(pseudo_label)
                    ds.update_image(path, org_split, pseudo_label)
                    pseudos += 1
                    
            print("First 10 probas:", probas[:10])

            print(f"Active Learning: {labeled} queried. Pseudos: {pseudos}")
            plot(features, top_n_idx, cluster_labels, dataset_info.name)

        except Exception:
            logging.error(traceback.format_exc())
        return ds



def plot(features, top_n_idx, cluster_labels, dataset_name):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))

    # Alle Punkte mit Clusterfarben
    num_clusters = len(np.unique(cluster_labels))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=cluster_labels,
        cmap='tab20',  # Oder 'tab10', je nach Anzahl
        alpha=0.6,
        label='Clusters'
    )

    # Top-N repräsentative Punkte markieren (z. B. mit schwarzem Rand)
    plt.scatter(
        features_2d[top_n_idx, 0], features_2d[top_n_idx, 1],
        facecolors='none',
        edgecolors='black',
        linewidths=1.5,
        s=80,
        label='Top-N Oracle'
    )

    plt.title(f'2D PCA of Unlabeled Features — {dataset_name}')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(*scatter.legend_elements(), title="Clusters", loc='upper right')
    plt.grid(True)

    plt.savefig(f"/workspace/Data-Centric-Image-Classification/images/{str(dataset_name)}_pca.png",
                bbox_inches='tight', dpi=300)
    plt.close()


def main(argv):
    alg = SemanticInitSpreading()
    alg.apply_algorithm()
    alg.report.show()

if __name__ == '__main__':
    app.run(main)