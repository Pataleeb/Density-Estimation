import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from scipy.stats import gaussian_kde


file_path="n90pol.csv"
data = pd.read_csv(file_path)
amygdala=data['amygdala'].dropna()
acc=data['acc'].dropna()


nbin_amygdala=int(len(amygdala)**(-1/3))
bin_amygdala = max(nbin_amygdala, 10)
sbin_amygdala = (max(amygdala) - min(amygdala)) / bin_amygdala
boundary_amygdala = np.arange(min(amygdala) - 0.001, max(amygdala), sbin_amygdala)

nbin_acc = int(len(acc) ** (-1 / 3))
bin_acc = max(nbin_acc, 10)
sbin_acc = (max(acc) - min(acc)) / bin_acc
boundary_acc = np.arange(min(acc) - 0.001, max(acc), sbin_acc)

#Optimal bandwidth
def optimal_bandwidth(data):
    sigma=np.std(data,ddof=1)
    m=len(data)
    return 1.06*sigma*(m**(-1/5))

bw_amygdala=optimal_bandwidth(amygdala)
bw_acc=optimal_bandwidth(acc)

# Plot histogram for Amygdala
plt.figure(figsize=(12, 5))
sns.histplot(amygdala, bins=boundary_amygdala, stat="density", color="blue", alpha=0.6)
plt.title("Histogram of Amygdala")
plt.xlabel("Amygdala ")
plt.ylabel("Density")
#plt.savefig("hamygdala.png")
plt.show()

# Plot KDE for Amygdala
plt.figure(figsize=(12, 5))
sns.kdeplot(amygdala, bw_adjust=bw_amygdala, color="red", label="KDE")
plt.title("KDE of Amygdala")
plt.xlabel("Amygdala ")
plt.ylabel("Density")
plt.legend()
#plt.savefig("kde_amygdala.png")
plt.show()

# Plot histogram for ACC
plt.figure(figsize=(12, 5))
sns.histplot(acc, bins=boundary_acc, stat="density", color="green", alpha=0.6)
plt.title("Histogram of ACC")
plt.xlabel("ACC")
plt.ylabel("Density")
#plt.savefig("hacc.png")
plt.show()

# Plot KDE for ACC
plt.figure(figsize=(12, 5))
sns.kdeplot(acc, bw_adjust=bw_acc, color="red", label="KDE")
plt.title("KDE of ACC")
plt.xlabel("ACC ")
plt.ylabel("Density")
plt.legend()
#plt.savefig("kdeacc.png")
plt.show()

###Q2 - 2D histogram

nbin= max(int(len(amygdala)**(-1/3)), 10)
hist, xedges, yedges = np.histogram2d(amygdala, acc, bins=[nbin,nbin], density=True)

xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])
xpos = xpos.flatten() / 2.0
ypos = ypos.flatten() / 2.0
zpos = np.zeros_like(xpos)

dx = (xedges[1] - xedges[0]) * np.ones_like(zpos)
dy = (yedges[1] - yedges[0]) * np.ones_like(zpos)
dz = hist.flatten()

dz_normalized = dz / dz.max()
colors = cm.viridis(dz_normalized)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

ax.set_xlabel('Amygdala')
ax.set_ylabel('ACC')
ax.set_zlabel('Density')
ax.set_title('2D Histogram of Amygdala vs ACC')
#plt.savefig("2d_histogram_amygdala_acc.png")
plt.show()

##Q3
file_path="n90pol.csv"
data = pd.read_csv(file_path)
amygdala=data['amygdala'].dropna().values
acc=data['acc'].dropna().values

kde = gaussian_kde(np.vstack([amygdala,acc]), bw_method='scott')

gridno = 100
xmin, xmax = np.percentile(amygdala, [1, 99])
ymin, ymax = np.percentile(acc, [1, 99])
X, Y = np.meshgrid(np.linspace(xmin, xmax, gridno), np.linspace(ymin, ymax, gridno))
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)


fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, levels=40, cmap="plasma")
ax.contour(X, Y, Z, levels=20, colors="black", linewidths=0.5)

ax.scatter(amygdala, acc, color="white", s=5, alpha=0.5, label="Data Points")

cbar = plt.colorbar(contour)
cbar.set_label("Density")

ax.set_xlabel("Amygdala")
ax.set_ylabel("ACC")
ax.set_title("2D KDE Contour Plot")
ax.legend()
#plt.savefig("2d_contour.png",dpi=300,bbox_inches="tight")

plt.show()

##################################

gridno = 100
xmin, xmax = np.percentile(amygdala, [1, 99])
ymin, ymax = np.percentile(acc, [1, 99])
X, Y = np.meshgrid(np.linspace(xmin, xmax, gridno), np.linspace(ymin, ymax, gridno))
positions = np.vstack([X.ravel(), Y.ravel()])


base_kde = gaussian_kde(np.vstack([amygdala, acc]), bw_method='scott')
base_bw = base_kde.factor


bw_factors = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, bw_factor in enumerate(bw_factors):

    adjusted_bw = base_bw * bw_factor
    kde = gaussian_kde(np.vstack([amygdala, acc]), bw_method=adjusted_bw)
    Z = np.reshape(kde(positions).T, X.shape)


    ax = axes[i]
    contour = ax.contourf(X, Y, Z, levels=40, cmap="plasma")
    ax.contour(X, Y, Z, levels=20, colors="black", linewidths=0.5)

    ax.scatter(amygdala, acc, color="white", s=5, alpha=0.5, label="Data Points")

    ax.set_xlabel("Amygdala")
    ax.set_ylabel("ACC")
    ax.set_title(f"2D KDE Contour (Bandwidth = Scott * {bw_factor})")

fig.colorbar(contour, ax=axes, orientation="vertical", shrink=0.8, label="Density")
#plt.savefig("2d_contour_tuned.png", dpi=300, bbox_inches="tight")
plt.show()

###Q4
###conditonal probability of amygdala|orientation
orientation_values = [2, 3, 4, 5]
cmeans={"Orientation":[],'Mean_amygdala':[],'Mean_acc':[]}
fig,axes = plt.subplots(2, 4, figsize=(16, 8))
bandwidth=1.0
for i, c in enumerate(orientation_values):
    subset_amy=data[data['orientation'] == c]['amygdala'].dropna()
    subset_acc = data[data['orientation'] == c]['acc'].dropna()


    mean_amy=subset_amy.mean()
    mean_acc=subset_acc.mean()

    cmeans["Orientation"].append(c)
    cmeans["Mean_amygdala"].append(mean_amy)
    cmeans["Mean_acc"].append(mean_acc)

    sns.kdeplot(subset_amy, bw_adjust=bandwidth, fill=True, ax=axes[0, i])
    axes[0, i].set_title(f"p(Amygdala | Orientation = {c})")
    axes[0, i].set_xlabel("Amygdala")
    axes[0, i].set_ylabel("Density")

    sns.kdeplot(subset_acc, bw_adjust=bandwidth, fill=True, ax=axes[1, i])
    axes[1, i].set_title(f"p(ACC | Orientation = {c})")
    axes[1, i].set_xlabel("ACC")
    axes[1, i].set_ylabel("Density")

plt.tight_layout()
#plt.savefig("conditional_kde_plot.png", dpi=300, bbox_inches="tight")

plt.show()

conditional_mean=pd.DataFrame(cmeans)
print(conditional_mean)

####Q5
def plot_conditional_kde(data,orientation_col,orientation_values,save_path="conditional_kde_plot.png"):
    x_col='amygdala'
    y_col='acc'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, c in enumerate(orientation_values):
        subset=data[data['orientation'] == c]
        kde=gaussian_kde(np.vstack([subset[x_col], subset[y_col]]), bw_method='scott')
        x_grid,y_grid=np.mgrid[subset[x_col].min():subset[x_col].max():100j,
                      subset[y_col].min():subset[y_col].max():100j]

        z=kde(np.vstack([x_grid.flatten(), y_grid.flatten()]))

        ax = axes[idx]
        contour = ax.contourf(x_grid, y_grid, z.reshape(x_grid.shape), cmap='viridis', alpha=0.6, levels=10)
        fig.colorbar(contour, ax=ax, label="Density")
        ax.set_title(f"P({x_col}, {y_col} | {orientation_col}={c})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    plt.tight_layout()
    #plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()

plot_conditional_kde(data, 'orientation', orientation_values,save_path="conditional_kde.png")