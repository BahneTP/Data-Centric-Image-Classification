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


class TestInitSpreading(AlgorithmSkelton):
    def __init__(self):
        name = "testdata_init_spreading"
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
            test_paths, _ = ds.get_training_subsets('test')
            all_paths, _ = ds.get_training_subsets('all')
            n_unlabeled = len(unlabeled_paths)
            n_all = len(all_paths)
            n_test = len(test_paths)
            n_init = int(1*n_unlabeled)
            n_active = 1-n_init

            nc = len(dataset_info.classes)  # Number of classes.
            p = 2                           # How often to label one image.
            k_nearest = 2#(1/0.25)//p
            k_clusters = nc*4               # Number of clusters for kmeans.

            # p_dict = {2: 2, 6: 2, 8:,1:}

            print(f'n_all: {n_all}, n_unlabeled: {n_unlabeled}, n_test: {n_test}')

    # 2. Initialisation
            budget=0                        # Counter.
            labeled_paths = []

        # 2.1 Extracting the features.
            features_all = self.extractFeatures(unlabeled_paths=all_paths)
            features_unlabeled = features_all[:n_unlabeled]
            features_test = features_all[n_unlabeled:n_unlabeled+n_test]

        # 2.2 Getting labels close to test data
            unlabeled_to_all = {path: idx for idx, path in enumerate(all_paths) if path in set(unlabeled_paths)}

            top_n_idx = set()

            for i, test_feat in enumerate(features_test):
                dists = np.linalg.norm(features_unlabeled - test_feat, axis=1)
                nearest = np.argsort(dists)[:k_nearest]
                for ni in nearest:
                    path = unlabeled_paths[ni]

                    if budget == n_unlabeled or path in labeled_paths:
                        continue
                    
                    abs_idx = unlabeled_to_all[path]
                    top_n_idx.add(abs_idx)
                    budget+=p
                    labeled_paths.append(path)
            print(f'budget after test-init: {budget/n_unlabeled}')

        # 2.3 KMeans Clustering
            # kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(features_all)
            # cluster_labels = kmeans.labels_
            # cluster_centers =  kmeans.cluster_centers_

        # 2.4 Using the last Budget
            k=k_nearest +4
            for i, test_feat in enumerate(features_test):
                if budget >= n_unlabeled:
                    break
                k+=1
                dists = np.linalg.norm(features_unlabeled - test_feat, axis=1)
                next_nearest = np.argsort(dists)[k:k+3]

                for ni in next_nearest:
                    path = unlabeled_paths[ni]
                    if path in labeled_paths:
                        continue
                    if p > n_unlabeled-budget and (p!=1):
                        break

                    abs_idx = unlabeled_to_all[path]
                    top_n_idx.add(abs_idx)
                    budget+=p
                    labeled_paths.append(path)

                    if budget >= n_unlabeled:
                        break

            print(f'budget after "Using the last Budget": {budget/n_unlabeled}')

        # 2.5 Calling the oracle.

            labeled_indices = []
            labeled_labels = []
            pseudos = 0
            oracle_count = 0
            labeled = 0
            for i in top_n_idx:
                path = all_paths[i]
                org_split = ds.get(path, 'original_split')
                oracle_label = [float(x) for x in oracle.get_soft_gt(path, p)]
                oracle_count += p
                ds.update_image(path, org_split, oracle_label)
                labeled += 1
                labeled_indices.append(i)
                labeled_labels.append(oracle_label)
            print(f'oracle_count: {oracle_count/n_unlabeled}')
    

    # 3. Method Here we skip the method part.


    # 4. Label Spreading
            int_labels = np.full(n_all, -1)
            for idx, lbl in zip(labeled_indices, labeled_labels):
                int_labels[idx] = int(np.argmax(lbl))


            label_spread = LabelSpreading(kernel='rbf', alpha=0.2, max_iter=40, gamma=0.01)
            label_spread.fit(features_all, int_labels)
            probas = label_spread.label_distributions_

            for i, path in enumerate(all_paths):
                if i not in labeled_indices:
                    org_split = ds.get(path, 'original_split')
                    pseudo_label = list(map(float, probas[i]))
                    # print(pseudo_label)
                    ds.update_image(path, org_split, pseudo_label)
                    pseudos += 1
                    
            print("First 10 probas:", probas[:10])

            print(f"Active Learning: {labeled} queried. Pseudos: {pseudos}")
            # plot(features_all, top_n_idx, cluster_labels, dataset_info.name)

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
    alg = TestInitSpreading()
    alg.apply_algorithm()
    alg.report.show()

if __name__ == '__main__':
    app.run(main)