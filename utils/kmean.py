import torch
class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=False,device = torch.device("cpu")):

        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")])
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0

    def fit(self, x, is_norm = True):
        if is_norm:
            x = x / x.norm(dim=1, keepdim=True)

        # 可以改為 kmean++ 的初始化方式改善收斂
        self.centers = x[torch.randperm(x.size(0))[:self.n_clusters]].clone()

        while True:
            # 標記具類
            self.nearest_center(x)
            # 更新中心點
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        centers = self.centers / self.centers.norm(dim=1, keepdim=True)
        dists = 1 - (x @ centers.t())
        self.labels = torch.argmin(dists, dim=1)
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1]), device=x.device)
        
        for i in range(self.n_clusters):
            centers = torch.cat([centers, torch.mean(x[self.labels==i], dim=0, keepdim=True)], dim=0)
        self.centers = centers

    def representative_sample(self):
        # 距離中心點最近的樣本
        self.representative_samples = torch.argmin(self.dists, dim=0)