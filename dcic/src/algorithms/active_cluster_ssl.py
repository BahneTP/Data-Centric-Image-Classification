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
from sklearn.cluster import KMeans
import math
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

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


class ActiveLearning(AlgorithmSkelton):
    def __init__(self):
        name = "active_cluster_ssl"
        AlgorithmSkelton.__init__(self, name)

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.eval()
        self.model = torch.nn.Sequential(
            model.features,
            torch.nn.AdaptiveAvgPool2d(1),
        )
        weights = EfficientNet_B0_Weights.DEFAULT
        self.transform = weights.transforms()

    def extractFeatures(self, unlabeled_paths):
        dataset = ImageDataset(unlabeled_paths, self.transform, "/workspace/Data-Centric-Image-Classification/raw_datasets/")
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        features = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                output = self.model(batch)
                output = output.view(output.size(0), -1)  # Robust flatten
                features.append(output.cpu().numpy())
                print(f'{i}/{len(loader)}')

        features = np.concatenate(features, axis=0)
        return features

    def run(self, ds, oracle, dataset_info, v_fold, num_annos, percentage_labeled):
        try:

            
    # 1. Setup
            nc = len(dataset_info.classes)  # Number of classes.
            p = 3                           # How often to label one image.
            k_clusters = nc*1               # Number of clusters for kmeans.
            n_query=50

            unlabeled_paths, _ = ds.get_training_subsets('unlabeled')
            test_paths, _ = ds.get_training_subsets('test')

    # 2. Initialisation
            
        # 2.1 Extracting the features.
            features = self.extractFeatures(unlabeled_paths=unlabeled_paths)

        # 2.2 KMeans Clustering
            kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(features)
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

        # 2.3 Grouping per cluster.
            cluster_to_indices = defaultdict(list)              #{1: [0, 1, 3], 2: [2, 4, 5]]
            for idx, label in enumerate(cluster_labels):
                cluster_to_indices[label].append(idx)


        # 2.4 Get central image per clusters.
            u = max(1, n_query // k_clusters)  # How many images per cluster
            top_n_idx = []

            for cluster_id, indices in cluster_to_indices.items():
                # Feature-Vektoren in diesem Cluster
                cluster_feats = features[indices]
                center = cluster_centers[cluster_id]
                dists = np.linalg.norm(cluster_feats - center, axis=1)

                # Sort, so that imamges close to the cluster-centers are at the beginning. (-> These are the ones we are aiming for.)
                sorted_idx = np.argsort(dists)
                selected = [indices[i] for i in sorted_idx[:u]]

                top_n_idx.extend(selected)

            print(f"Pro Cluster repräsentative Bilder ausgewählt: {len(top_n_idx)} insgesamt.")

            pseudos=0
            labeled=0

            for i, path in enumerate(unlabeled_paths):
                org_split = ds.get(path, 'original_split')
                if i in top_n_idx:
                    oracle_label = [float(x) for x in oracle.get_soft_gt(path, p)]
                    ds.update_image(path, org_split, oracle_label)
                    labeled += 1

    # 3. Method

                else:
                    distances = np.linalg.norm(cluster_centers - features[i], axis=1)
                    similarities = 1 / (distances + 1e-8)
                    pseudo_label = similarities / np.sum(similarities)
                    ds.update_image(path, org_split, pseudo_label.tolist())
                    pseudos += 1


##############################################TODO
            test = 0
            for path in test_paths:
                split = ds.get(path, 'original_split')
                if split == "test":
                    ds.update_image(path, split, nc * [0])
                    test += 1

            print(f"Active Learning: {labeled} queried. Test: {test}. Pseudos: {pseudos}")
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
    alg = ActiveLearning()
    alg.apply_algorithm()
    alg.report.show()

if __name__ == '__main__':
    app.run(main)