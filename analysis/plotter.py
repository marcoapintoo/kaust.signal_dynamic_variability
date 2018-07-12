import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = [18, 10]
sns.set_style("whitegrid")

COLOR_MAP = sns.diverging_palette(10, 220, sep=100, n=10, as_cmap=True)
COLOR_MAP = sns.light_palette((240, 75, 50), input="husl", reverse=True)

class Plotter:
    #EXTENSION = "jpg"
    EXTENSION = "png"
    def __init__(self, filename_template, basepath):
        self.basepath = basepath
        self.filename_template = filename_template
        if not os.path.exists(basepath):
            os.makedirs(basepath)
    
    def _save_figure(self, codesubject, codename):
        plt.savefig("{0}/{1}.{2}".format(
            self.basepath,
            self.filename_template.format(codesubject=codesubject, codename=codename),
            self.EXTENSION,
        ), dpi=150, transparent=True)
        [(plt.cla(), plt.clf(), plt.close()) for _ in range(4)]
    
    def _save_figure(self, codesubject, codename):
        try:
            plt.savefig("{0}/{1}.{2}".format(
                self.basepath,
                self.filename_template.format(codesubject=codesubject, codename=codename),
                self.EXTENSION,
            ), dpi=150, transparent=True)
        except Exception as identifier:
            print("EXCEPTION AT SAVING FIGURE", identifier)
        finally:
            [(plt.cla(), plt.clf(), plt.close()) for _ in range(4)]

    def plot(self, conditional):
        raise NotImplementedError()

class HistogramPlotter(Plotter):
    def __init__(self, filename_template="{codesubject}_{codename}_histogram", basepath="./plots/histograms"):
        super().__init__(filename_template, basepath)
    
    def plot(self, codesubject, conditional):
        print(":: Plotting the histogram")
        keynames = conditional.histograms.keys()        
        codenames = [name.strip().replace(" ", "_").lower() for name in keynames]
        for codename, keyname in zip(codenames, keynames):
            print("::: Codename", keyname)
            data_hist = pd.DataFrame(
                np.hstack([
                    conditional.bins[:-1].astype("i").reshape(-1, 1),
                    conditional.get_distribution(keyname, normalized=True).reshape(-1, 1),
                ]),
                columns=["LSH/TVVAR state", "Normalized count"]
            )
            data_hist["LSH/TVVAR state"] = data_hist["LSH/TVVAR state"].astype("i").astype("category")
            sns.barplot(x="LSH/TVVAR state", y="Normalized count", data=data_hist, ci=None)
            #plt.title("Empirical conditional distribution of {0}".format(keyname), fontsize="x-large")
            plt.title("Subject{0}. Empirical conditional distribution of {1}".format(codesubject, keyname), fontsize="x-large")
            if len(data_hist["LSH/TVVAR state"]) > 200:
                plt.xticks([])
            self._save_figure(codesubject, codename)

class SimpleDistanceHistogramPlotter(Plotter):
    def __init__(self, filename_template="{codesubject}_histogram_distance", basepath="./plots/distance_matrices"):
        super().__init__(filename_template, basepath)
    
    def plot(self, codesubject, conditional):
        print(":: Plotting the distance matrix of the histograms")
        plt.figure(figsize=(10, 10))
        keynames = conditional.histograms.keys()
        histogram_df = pd.DataFrame(
            conditional.distance_matrix,
            columns=keynames,
            index=keynames,
        )
        #print(histogram_df)
        sns.heatmap(histogram_df, vmax=0.5, cmap=COLOR_MAP, cbar=True)
        plt.title("Subject {0}".format(codesubject), fontsize="x-large")
        self._save_figure(codesubject, "")


class DistanceHistogramPlotter(Plotter):
    def __init__(self, filename_template="{codesubject}_histogram_distance", basepath="./plots/distance_matrices", label_groups=None, tight_mode=False):
        super().__init__(filename_template, basepath)
        self.label_groups = label_groups
        self.tight_mode = tight_mode
        self.m = 200
        self.default_title = "Subject {codesubject}"
        self.cmap = COLOR_MAP
        
    def plot_grid(self, codename, codesubjects, conditionals, rows=None, cols=None):
        if rows is None:
            rows = np.ceil(len(conditionals)/cols).astype("i")
        fig = plt.figure(figsize=(cols * 4, rows * 4.2))
        for m, (codesubject, conditional) in enumerate(zip(codesubjects, conditionals)):
            ax = fig.add_subplot(rows, cols, m + 1)
            self._plot(codesubject, conditional, ax)
        plt.tight_layout()
        self._save_figure(codename, "")
    
    def plot(self, codesubject, conditional):
        print(":: Plotting the distance matrix of the histograms", self.filename_template.format(codesubject=codesubject))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        self._plot(codesubject, conditional, ax)
        plt.tight_layout()
        self._save_figure(codesubject, "")
    
    def _plot(self, codesubject, conditional, ax):
        order = np.arange(conditional.distance_matrix.shape[0])
        groups = None
        coord_clusters = []
        if self.label_groups is not None:
            groups = np.array([np.array([conditional.keynames.index(v) for v in y]) for y in self.label_groups])
            order = np.concatenate(groups)
            coord_clusters = [0]
            for i, u in enumerate(groups):
                coord_clusters.append(coord_clusters[-1] + len(u))
        keynames = np.array(conditional.keynames)
        M = conditional.distance_matrix
        if np.all(np.isnan(M)):
            M = np.zeros(M.shape)
        histogram_df = pd.DataFrame(
            M[order, :][:, order],
            columns=keynames[order],
            index=keynames[order],
        )
        ax = sns.heatmap(histogram_df, vmax=self.m, cmap=self.cmap, cbar=not self.tight_mode, ax=ax)
        for i in range(0, len(coord_clusters) - 1):
            plt.plot( (coord_clusters[i+1], coord_clusters[i+1]), (coord_clusters[i], coord_clusters[i+1]), "w-", linewidth=10.0)
            plt.plot( (coord_clusters[i], coord_clusters[i+1]), (coord_clusters[i+1], coord_clusters[i+1]), "w-", linewidth=10.0)
            plt.plot( (coord_clusters[i], coord_clusters[i]), (coord_clusters[i], coord_clusters[i+1]), "w-", linewidth=10.0)
            plt.plot( (coord_clusters[i], coord_clusters[i+1]), (coord_clusters[i], coord_clusters[i]), "w-", linewidth=10.0)
            plt.plot( (coord_clusters[i+1], coord_clusters[i+1]), (coord_clusters[i], coord_clusters[i+1]), "r-", linewidth=3.5)
            plt.plot( (coord_clusters[i], coord_clusters[i+1]), (coord_clusters[i+1], coord_clusters[i+1]), "r-", linewidth=3.5)
            plt.plot( (coord_clusters[i], coord_clusters[i]), (coord_clusters[i], coord_clusters[i+1]), "r-", linewidth=3.5)
            plt.plot( (coord_clusters[i], coord_clusters[i+1]), (coord_clusters[i], coord_clusters[i]), "r-", linewidth=3.5)
        ax.axhline(y=0, color='k',linewidth=2)
        ax.axhline(y=conditional.distance_matrix.shape[1], color='k',linewidth=2)
        ax.axvline(x=0, color='k',linewidth=2)
        ax.axvline(x=conditional.distance_matrix.shape[0], color='k',linewidth=2)
        if self.tight_mode:
            plt.xticks([])
            plt.yticks([])
        #plt.title("Subject {0}".format(codesubject), fontsize="x-large")
        plt.title(self.default_title.format(codesubject=codesubject), fontsize="x-large")
        
        


class HistogramMatrixPlotter(Plotter):
    def __init__(self, filename_template="{codesubject}_histogram_matrix", basepath="./plots/histogram_matrix", label_groups=None, tight_mode=False):
        super().__init__(filename_template, basepath)
        self.label_groups = label_groups
        self.tight_mode = tight_mode
    
    def plot(self, codesubject, conditional):
        rows, cols = len(self.label_groups), max([len(c) for c in self.label_groups])
        fig = plt.figure(figsize=(cols*4, rows*4.2))
        for i, group in enumerate(self.label_groups):
            for j, keyname in enumerate(group):
                ax = fig.add_subplot(rows, cols, i * cols + j + 1)
                print("::: Codename", keyname)
                data_hist = pd.DataFrame(
                    np.hstack([
                        conditional.bins[:-1].astype("i").reshape(-1, 1),
                        conditional.get_distribution(keyname, normalized=True).reshape(-1, 1),
                    ]),
                    columns=["LSH/TVVAR state", "Normalized count"]
                )
                #print(data_hist)
                data_hist["LSH/TVVAR state"] = data_hist["LSH/TVVAR state"].astype("i").astype("category")
                sns.barplot(x="LSH/TVVAR state", y="Normalized count", data=data_hist, ci=None, ax=ax)
                plt.ylim(0, 0.15)
                plt.title("{0}: {1}".format(codesubject, keyname), fontsize="x-large")
                if self.tight_mode:
                    plt.xticks([])
                    #plt.yticks([])
        plt.title("Subject {0}".format(codesubject), fontsize="x-large")
        plt.tight_layout()
        self._save_figure(codesubject, "")
