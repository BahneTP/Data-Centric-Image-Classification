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
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from scipy.stats import entropy  # KL divergence


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


class ClusterInit(AlgorithmSkelton):
    def __init__(self):
        name = "cluster_init"
        AlgorithmSkelton.__init__(self, name)

        # ##### ResNet50
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        self.model = torch.nn.Sequential(
            *(list(model.children())[:-1])  # Bis einschließlich avgpool
        )
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
            unlabeled_paths, _ = ds.get_training_subsets('unlabeled')
            n = len(unlabeled_paths)
            n_init = int(0.5*n)
            test_paths, _ = ds.get_training_subsets('test')

            nc = len(dataset_info.classes)      # Number of classes.
            p = 5#math.ceil(nc/2)                 # How often to label one image.
            k_clusters = 2#nc*5                   # Number of clusters for kmeans.

            n_query=n_init // p

            print(f'n_query: {n_query}')

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

            pseudos=0
            labeled=0
            oracle_count=0

            for i, path in enumerate(unlabeled_paths):
                org_split = ds.get(path, 'original_split')
                if i in top_n_idx:
                    oracle_label = [float(x) for x in oracle.get_soft_gt(path, p)]
                    oracle_count+=p
                    ds.update_image(path, org_split, oracle_label)
                    labeled += 1
             


            # Features, cluster_labels, cluster_centers sind schon berechnet

            # Masken für Label-Status
            labeled_mask = np.zeros(n, dtype=bool)
            labeled_mask[top_n_idx] = True

            # Dummy-Labels initialisieren (du hast ja erst nur die aus top_n_idx)
            labels = np.zeros((n, nc))
            for i, path in enumerate(unlabeled_paths):
                if labeled_mask[i]:
                    labels[i] = ds.get(path, 'labels')  # Passe an, wie du ans Label kommst

            budget = int(n * 0.5)  # z.B. voll ausreizen
            query_batch_size = 5   # Pro Cluster pro Runde (nach Wunsch erhöhen)
            max_iters = 1#int((budget - np.sum(labeled_mask)) // (k_clusters * query_batch_size))
            print("maxiters")
            print(max_iters)
            
            skips = 0
            for al_iter in range(max_iters):
                # 1. Trainiere Klassifikator auf aktuellen Labels (nur die, die gelabelt wurden)
                X_labeled = features[labeled_mask]
                y_labeled = labels[labeled_mask].argmax(axis=1)
                if len(np.unique(y_labeled)) < 2:
                    skips+=1
                    continue

                clf = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')
                clf.fit(X_labeled, y_labeled)

                # 2. Pro Cluster: wähle das Sample mit größter KL
                query_indices = []
                for cluster_id, indices in cluster_to_indices.items():
                    # Nur ungelabelte im Cluster
                    idxs_unlabeled = [i for i in indices if not labeled_mask[i]]
                    if len(idxs_unlabeled) == 0:
                        continue
                    # Probs für Samples
                    X_unlabeled = features[idxs_unlabeled]
                    probs_samples = clf.predict_proba(X_unlabeled)
                    # Prototyp-Feature (= Clusterzentrum) durch das Modell schicken:
                    proto_prob = clf.predict_proba(cluster_centers[cluster_id].reshape(1,-1))[0]  # Shape (nc,)
                    # KL-Divergenz für alle im Cluster
                    kls = [entropy(ps, proto_prob) for ps in probs_samples]  # KL(P_sample || P_proto)
                    # Index mit höchster KL
                    topk = np.argsort(kls)[-query_batch_size:]
                    to_query = [idxs_unlabeled[i] for i in topk]
                    query_indices.extend(to_query)

                # 3. Oracle abfragen
                print(0)
                for i in query_indices:
                    print(1)
                    path = unlabeled_paths[i]
                    org_split = ds.get(path, 'original_split')
                    oracle_label = [float(x) for x in oracle.get_soft_gt(path, p)]
                    ds.update_image(path, org_split, oracle_label)
                    labels[i] = oracle_label
                    labeled_mask[i] = True

                print(f"Active-Labeling Iteration {al_iter+1}, Queried: {len(query_indices)} new, Total labeled: {labeled_mask.sum()}, skips: {skips}")

                # Optional: Budget prüfen/abbrechen falls voll
                if labeled_mask.sum() >= budget:
                    break
            print("skips")
            print(skips)




            # for i, path in enumerate(unlabeled_paths):
            #     if (oracle_count/n)<1.0:       
            #         if not i in top_n_idx:
            #             if (n-oracle_count) < p:
            #                 p=1
            #             oracle_label = [float(x) for x in oracle.get_soft_gt(path, p)]
            #             oracle_count+=p
            #             ds.update_image(path, org_split, oracle_label)
            #             labeled += 1
            #     else:
            #         break



    # 3. Method Here we skip the method part.



##############################################
            test = 0
            for path in test_paths:
                split = ds.get(path, 'original_split')
                if split == "test":
                    ds.update_image(path, split, nc * [0])
                    test += 1

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
    alg = ClusterInit()
    alg.apply_algorithm()
    alg.report.show()

if __name__ == '__main__':
    app.run(main)