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
from torchvision.models import ResNet18_Weights
from sklearn.cluster import KMeans
import math


from torch.utils.data import Dataset, DataLoader

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
        name = "active_learning"
        AlgorithmSkelton.__init__(self, name)

        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.eval()
        self.model = torch.nn.Sequential(*list(model.children())[:-1])  # bis avgpool

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def run(self, ds, oracle, dataset_info, v_fold, num_annos, percentage_labeled):
        try:
            pseudos=0
            labeled=0
            
            num_images = len(ds.get_training_subsets('unlabeled')[0])
            
            k = len(dataset_info.classes)  # Anzahl Klassen
            p = max(2, math.ceil(1 * math.log2(k)))           # Wie oft top n Bild labeln.
            n_query = math.floor(num_images / p)    # wie viele Bilder aktiv labeln

            print(f'k: {k}, p: {p}, n_query: {n_query}\n')

            paths, _ = ds.get_training_subsets('all')

            features = []
            unlabeled_paths = []

        ################ Getting the Features ################
            for path in paths:
                split = ds.get(path, 'split')
                if split == "unlabeled":
                    unlabeled_paths.append(path)

            print(f'Starting extracting features with DataLoader.\n')
            dataset = ImageDataset(unlabeled_paths, self.transform, "/workspace/Data-Centric-Image-Classification/raw_datasets/")
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

            features = []

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            with torch.no_grad():
                for i, batch in enumerate(loader):
                    batch = batch.to(device)
                    output = self.model(batch).squeeze()
                    features.append(output.cpu().numpy())
                    print(f'{i}/{len(loader)}')

            features = np.concatenate(features, axis=0)

            cluster_center = np.mean(features, axis=0)
            dists = np.linalg.norm(features - cluster_center, axis=1)

            top_n_idx = np.argsort(-dists)[:n_query]
            

            # Clustering der Features
            k_clusters = k  # Anzahl Klassen
            kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(features)
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            for i, path in enumerate(unlabeled_paths):
                org_split = ds.get(path, 'original_split')
                if i in top_n_idx:
                    oracle_label = [float(x) for x in oracle.get_soft_gt(path, p)]
                    ds.update_image(path, org_split, oracle_label)
                    labeled += 1
                else:

        # ECHTES Pseudo-Labeling: Soft Label basierend auf Distanz zu allen Zentren
                    distances = np.linalg.norm(cluster_centers - features[i], axis=1)
                    similarities = 1 / (distances + 1e-8)
                    pseudo_label = similarities / np.sum(similarities)
                    ds.update_image(path, org_split, pseudo_label.tolist())
                    pseudos += 1

            # test: nur Dummy
            test = 0
            for path in paths:
                # split = ds.get(path, 'split')
                split = ds.get(path, 'original_split')
                if split == "test":
                    ds.update_image(path, split, k * [0])
                    test += 1

            print(f"Active Learning: {labeled} queried. Test: {test}. Pseudos: {pseudos}")
            plot(features, top_n_idx, dataset_info.name)

        except Exception:
            logging.error(traceback.format_exc())
        return ds

def plot(features, top_n_idx, dataset_name):
    # PCA-Plot für Debugging
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c='lightgray', label='Pseudo-labeled'
    )
    plt.scatter(
        features_2d[top_n_idx, 0], features_2d[top_n_idx, 1],
        c='red', label='Top-N Oracle'
    )
    plt.title('2D PCA of Unlabeled Features')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid(True)
    # plt.show()

    plt.savefig(f"/workspace/Data-Centric-Image-Classification/images/{str(dataset_name)}_pca.png", bbox_inches='tight', dpi=300)
    plt.close()

def main(argv):
    alg = ActiveLearning()
    alg.apply_algorithm()
    alg.report.show()

if __name__ == '__main__':
    app.run(main)