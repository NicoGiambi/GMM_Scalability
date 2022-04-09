import itertools
import json
import math
import numpy as np
import seaborn as sns
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from copy import deepcopy

dataset_folder = "../datasets/"
plots_folder = "plots/"

with open(f'{dataset_folder}wh.txt', 'r') as f:
    wh = json.load(f)

with open(f'{dataset_folder}hw.txt', 'r') as f:
    hw = json.load(f)

with open(f'{dataset_folder}area.txt', 'r') as f:
    area = json.load(f)

w = []
h = []

for cat in hw:
    for i in range(len(wh[cat])):
        h.append(math.sqrt(area[cat][i] * hw[cat][i]))
        w.append(math.sqrt(area[cat][i] / hw[cat][i]))

# with open("width.txt", "w") as outfile:
#     outfile.write("\n".join(str(item) for item in w))
#
# with open("height.txt", "w") as outfile:
#     outfile.write("\n".join(str(item) for item in h))
#
# with open("dataset.txt", "w") as outfile:
#     outfile.write("\n".join(f"{str(w[i])} {str(h[i])}" for i in range(len(w))))


color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


def make_plots(gmm_mu, gmm_sigma, k=5):
    gmm_mu = np.array(gmm_mu)
    gmm_sigma = np.array(gmm_sigma)

    plot_results(points, indices, gmm_mu, gmm_sigma, 'clusters')

    plt.hist2d(x=np_w, y=np_h, bins=100)
    plt.show()
    plt.figure(figsize=(32, 20))
    plt.scatter(x=w, y=h, s=0.1)
    plt.savefig(f'{plots_folder}hwscatter.png')
    plt.show()

    plt.hist(np.clip(np_w / np_h, 0, 6), bins=200, alpha=0.5, color='red')
    plt.hist(np.clip(np_h / np_w, 0, 6), bins=200, alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 10))
    print(gmm_mu)
    for m in gmm_mu:
        ax = plt.gca()
        rect = mpl.patches.Rectangle((-m[0] / 2, -m[1] / 2), m[0], m[1], fill=False)
        ax.add_patch(rect)

    plt.plot(-200, -200)
    plt.plot(-200, 200)
    plt.plot(200, -200)
    plt.plot(200, 200)

    plt.show()

    plt.figure(figsize=(10, 10))

    kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
    centers = kmeans.cluster_centers_
    print(centers)

    for i, m in enumerate(centers):
        ax = plt.gca()
        rect = mpl.patches.Rectangle((-m[0] / 2, -m[1] / 2), m[0], m[1], fill=False)
        ax.add_patch(rect)

    plt.plot(-200, -200)
    plt.plot(-200, 200)
    plt.plot(200, -200)
    plt.plot(200, 200)

    plt.show()

    plt.figure(figsize=(10, 10))

    pgmm = GaussianMixture(n_components=k, covariance_type='full', max_iter=100, random_state=0, tol=1e-5)
    pgmm.means_init = centers
    pgmm.fit(points)
    centers = pgmm.means_
    print(centers)

    for i, m in enumerate(centers):
        ax = plt.gca()
        rect = mpl.patches.Rectangle((-m[0] / 2, -m[1] / 2), m[0], m[1], fill=False)
        ax.add_patch(rect)

    plt.plot(-200, -200)
    plt.plot(-200, 200)
    plt.plot(200, -200)
    plt.plot(200, 200)

    plt.show()


def plot_results(X, Y_, means, covariances, title):
    ax = plt.gca()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color='red')
        ell.set_alpha(0.5)
        ax.add_patch(ell)

    plt.title(title)
    plt.show()


def augment(extra=16, ann=''):
    new_points = list(deepcopy(points))
    for i, p in enumerate(points):
        x = p[0]
        y = p[1]
        print(f"\rCreating point:{i}/{len(points)}", end="")
        for j in range(extra):
            r = np.deg2rad(np.random.randint(0, 360))
            x_new = x + (x * np.random.uniform(0.0001, 0.1) * np.cos(r))
            y_new = y + (y * np.random.uniform(0.0001, 0.1) * np.sin(r))
            new_points.append([x_new, y_new])

    print(len(points))
    print(len(new_points))

    np.save(f"{dataset_folder}dataset_{extra}{ann}.npy", np.array(new_points))


def augment_and_show(extra=16, ann=''):
    augment(extra, ann)
    plt.figure(figsize=(20, 20))
    ds = np.load(f'{dataset_folder}dataset_{extra}{ann}.npy').transpose()
    plt.scatter(ds[0], ds[1], s=0.05)
    plt.savefig(f'{plots_folder}augmented_{extra}{ann}_scatter.png')
    plt.show()
    bins = 100
    plt.hist2d(ds[0], ds[1], bins=bins)
    plt.savefig(f'{plots_folder}augmented_{extra}{ann}_hist2d_{bins}.png')
    plt.show()

with open("../model/GMM/predictions/part-00000", "r") as f:
    indices = f.read().split('\n')

gmm_weights = [0.378064, 0.239208, 0.382728]
gmm_mu = [[56.5030901136114, 65.1273767851942],
          [199.1742698398557, 192.21783049652925],
          [16.42360617824805, 21.045464723375765]]
gmm_sigma = [[[847.4839598367946, 281.0034521252157],
              [281.0034521252157, 1320.2649658788757]],
             [[15364.620616868875, 5084.102899259593],
              [5084.102899259593, 11039.208619379386]],
             [[87.81482155694289, 52.28174038200124],
              [52.28174038200124, 148.83018480171694]]]

np_w = np.array(w)
np_h = np.array(h)
points = np.zeros((2, np_h.shape[0]))
points[0] = np_w
points[1] = np_h
points = points.transpose()
indices = np.array([int(el) for el in indices[:-1]])

make_plots(gmm_mu, gmm_sigma)

extra = 2
ann = '_random'

augment_and_show(extra, ann)

ds = np.load(f'{dataset_folder}dataset_{extra}{ann}.npy')
with open(f"{dataset_folder}dataset_{extra}{ann}.txt", "w") as outfile:
    outfile.write("\n".join(f"{str(ds[i,0])} {str(ds[i,1])}" for i in range(ds.shape[0])))#
