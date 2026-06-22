import torch
import numpy as np
import sklearn.manifold
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.collections
import mpl_toolkits.mplot3d.art3d

class IGNGVisualization:

    def __init__(
        self,
        igng,
        device=None,
        node_size=4,
        edge_width=0.2,
        edge_alpha=0.15
    ):

        self.device = (
            device
            if device is not None
            else getattr(igng, "device", "cpu")
        )

        print(
            "🛈 Visualization on GPU"
            if self.device == "cuda"
            else "🛈 Visualization on CPU"
        )

        # =====================================================
        # GRAPH
        # =====================================================

        self.nodes = igng.nodes.to(self.device)

        # format:
        # {node: {neighbor: age}}
        self.edges = igng.edges

        # optional labels
        self.labels = getattr(igng, "labels", None)

        if self.labels is not None:

            self.labels = self.labels.to(self.device)

            self.colors = [
                self.labels[i].argmax().item()
                for i in range(self.nodes.shape[0])
            ]

        else:

            self.colors = "blue"

        self.node_size = node_size
        self.edge_width = edge_width
        self.edge_alpha = edge_alpha
        self.data_colors = None
        self.data_colors = None
        self.current_colors = None

    # =====================================================
    # EDGE INDEX
    # =====================================================

    def _edge_index(self):

        src = []
        dst = []

        n_nodes = self.nodes.shape[0]

        for i, neighbors in self.edges.items():

            # sécurité clé invalide
            if i is None:
                continue

            if not isinstance(i, int):
                continue

            if i >= n_nodes:
                continue

            if neighbors is None:
                continue

            for j in neighbors.keys():

                # sécurité voisin invalide
                if j is None:
                    continue

                if not isinstance(j, int):
                    continue

                if j >= n_nodes:
                    continue

                # éviter doublons
                if i < j:

                    src.append(i)
                    dst.append(j)

        return (
            torch.tensor(src, device=self.device),
            torch.tensor(dst, device=self.device)
        )

    # =====================================================
    # CORE PLOT
    # =====================================================

    def set_data_labels(self, labels):

        labels = labels.detach().cpu()

        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        self.original_data_colors = labels.numpy()

        self.data_colors = self.original_data_colors.copy()

    def _plot(
        self,
        projected,
        title,
        labels,
        save_path=None,
        third_dim=False,
        data_points=None
    ):

        projected = projected.detach().cpu()

        plt.figure(figsize=(12, 8))

        if data_points is not None:

            data_points = data_points.detach().cpu()

            if not third_dim:

                plt.scatter(
                    data_points[:, 0],
                    data_points[:, 1],
                    s=1,
                    c=self.current_colors,
                    alpha=0.3,
                    zorder=0
                )
            else:

                ax = plt.axes(projection="3d")

                ax.scatter3D(
                    data_points[:, 0],
                    data_points[:, 1],
                    data_points[:, 2],
                    s=1,
                    c="lightgray",
                    alpha=0.15,
                    zorder=0
                )

        # =================================================
        # NODES
        # =================================================

        if not third_dim:

            plt.scatter(
                projected[:, 0],
                projected[:, 1],
                s=self.node_size,
                c=self.colors,
                cmap="tab10",
                zorder=2
            )


            # =================================================
            # LABELS DES NOEUDS IGNG
            # =================================================

            if self.labels is not None:

                for i in range(len(projected)):

                    plt.text(
                        projected[i,0] + 0.8,
                        projected[i,1] + 0.8,
                        f"{i}:{self.colors[i]}",
                        fontsize=8,
                        fontweight="bold",
                        zorder=5
                    )

        else:

            ax = plt.axes(projection="3d")

            ax.scatter3D(
                projected[:, 0],
                projected[:, 1],
                projected[:, 2],
                s=self.node_size,
                c=self.colors,
                cmap="tab10",
                zorder=2
            )

        # =================================================
        # EDGES
        # =================================================

        src, dst = self._edge_index()

        if src.numel() > 0:

            edges = torch.stack([
                projected[src.cpu()],
                projected[dst.cpu()]
            ])

            edges = edges.numpy().transpose(1, 0, 2)

            if third_dim:

                lc = mpl_toolkits.mplot3d.art3d.Line3DCollection(
                    edges,
                    linewidths=self.edge_width,
                    colors=(0.5, 0.5, 0.5, self.edge_alpha),
                    zorder=1
                )

                ax.add_collection(lc)

            else:

                lc = matplotlib.collections.LineCollection(
                    edges,
                    linewidths=self.edge_width,
                    colors=(0.5, 0.5, 0.5, self.edge_alpha),
                    zorder=1
                )

                plt.gca().add_collection(lc)

        # =================================================
        # LABELS
        # =================================================

        plt.title(title)

        plt.xlabel(labels[0])

        plt.ylabel(labels[1])

        if third_dim:
            ax.set_zlabel(labels[2])

        # =================================================
        # LEGEND
        # =================================================

        if self.labels is not None:

            unique = sorted(set(self.colors))

            handles = [

                plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    markerfacecolor=plt.cm.tab10(i / 10),
                    label=str(u),
                    markersize=8
                )

                for i, u in enumerate(unique)
            ]

            plt.legend(
                handles=handles,
                title="Classes"
            )

        # =================================================
        # SAVE
        # =================================================

        if save_path is not None:

            plt.savefig(
                save_path,
                bbox_inches="tight",
                dpi=300
            )

            print(f"✓ Saved visualization: {save_path}")

        plt.show()

# =====================================================
# PCA
# =====================================================

    def pca(
        self,
        data=None,
        third_dim=False,
        max_points=10000,
        save_path=None
    ):

        print("⏲ PCA...")

        x_nodes = self.nodes


        # =========================================
        # DATA MNIST
        # =========================================

        if data is not None:

            data = data.to(self.device)

            if len(data) > max_points:

                idx = torch.randperm(len(data))[:max_points]

                data = data[idx]


                # synchroniser couleurs
                if self.data_colors is not None:

                    self.data_colors = self.original_data_colors[
                        idx.cpu().numpy()
                    ]


            full = torch.cat(
                [
                    x_nodes,
                    data
                ],
                dim=0
            )

        else:

            full = x_nodes



        # =========================================
        # PCA
        # =========================================

        full = full - full.mean(dim=0)


        U, S, V = torch.pca_lowrank(full)


        proj = full @ (
            V[:, :3]
            if third_dim
            else V[:, :2]
        )



        # séparation

        proj_nodes = proj[:len(x_nodes)]

        proj_data = None

        if data is not None:

            proj_data = proj[len(x_nodes):]

        if data is not None:
            if self.data_colors is not None:
                self.current_colors = self.data_colors[:len(data)]

            else:
                self.current_colors = "gray"

        # =========================================
        # PLOT
        # =========================================

        self._plot(
            proj_nodes,
            "PCA Projection",
            ("PC1","PC2","PC3")
            if third_dim
            else ("PC1","PC2"),
            save_path,
            third_dim,
            data_points=proj_data
        )

    # =====================================================
    # TSNE
    # =====================================================

    def tsne(
        self,
        data=None,
        third_dim=False,
        max_points=5000,
        save_path=None
    ):

        print("⏲ t-SNE...")

        x_nodes = self.nodes

        # -------------------------------------------------
        # Optional dataset
        # -------------------------------------------------

        if data is not None:

            data = data.to(self.device)

            if len(data) > max_points:

                idx = torch.randperm(len(data))[:max_points]

                data = data[idx]

                # synchroniser les couleurs
                if self.data_colors is not None:

                    self.data_colors = self.original_data_colors[
                        idx.cpu().numpy()
                    ]

            full = torch.cat(
                [
                    x_nodes,
                    data
                ],
                dim=0
            )

        else:

            full = x_nodes


        # -------------------------------------------------
        # TSNE
        # -------------------------------------------------

        full_np = full.detach().cpu().numpy()


        proj = sklearn.manifold.TSNE(
            n_components=3 if third_dim else 2,
            perplexity=min(10, len(full_np)-1),
            random_state=42
        ).fit_transform(full_np)


        proj = torch.tensor(proj)


        # split nodes/data

        proj_nodes = proj[:len(x_nodes)]

        proj_data = None

        if data is not None:

            proj_data = proj[len(x_nodes):]

        if data is not None:
            if self.data_colors is not None:
                self.current_colors = self.data_colors[:len(data)]

            else:
                self.current_colors = "gray"    


        # -------------------------------------------------
        # Plot
        # -------------------------------------------------

        self._plot(
            proj_nodes,
            "t-SNE Projection",
            ("t1","t2","t3")
            if third_dim
            else ("t1","t2"),
            save_path,
            third_dim,
            data_points=proj_data
        )

    # =====================================================
    # UMAP
    # =====================================================

    def umap(
        self,
        data=None,
        third_dim=False,
        max_points=5000,
        save_path=None
    ):

        print("⏲ UMAP...")

        import umap

        x_nodes = self.nodes


        # =========================================
        # DATA MNIST
        # =========================================

        if data is not None:

            data = data.to(self.device)

            if len(data) > max_points:

                idx = torch.randperm(len(data))[:max_points]

                data = data[idx]


                # synchroniser couleurs
                if self.data_colors is not None:

                    self.data_colors = self.original_data_colors[
                        idx.cpu().numpy()
                    ]


            full = torch.cat(
                [
                    x_nodes,
                    data
                ],
                dim=0
            )

        else:

            full = x_nodes



        # =========================================
        # UMAP
        # =========================================

        x_np = (
            full.detach()
            .cpu()
            .numpy()
        )


        proj = umap.UMAP(
            n_components=3 if third_dim else 2,
            metric="euclidean",
            n_neighbors=min(15, len(x_np)-1),
            random_state=42
        ).fit_transform(x_np)


        proj = torch.tensor(proj)



        # séparation

        proj_nodes = proj[:len(x_nodes)]

        proj_data = None

        if data is not None:

            proj_data = proj[len(x_nodes):]

        if data is not None:
            if self.data_colors is not None:
                self.current_colors = self.data_colors[:len(data)]

            else:
                self.current_colors = "gray"    



        # =========================================
        # PLOT
        # =========================================

        self._plot(
            proj_nodes,
            "UMAP Projection",
            ("u1","u2","u3")
            if third_dim
            else ("u1","u2"),
            save_path,
            third_dim,
            data_points=proj_data
        )

    # =====================================================
    # DIRECT
    # =====================================================

    def direct(
        self,
        save_path=None
    ):

        if self.nodes.shape[1] > 3:

            raise ValueError(
                "Direct visualization requires <= 3 dimensions"
            )

        third_dim = (
            self.nodes.shape[1] == 3
        )

        self._plot(
            self.nodes,
            "Direct Space",
            ("x", "y", "z")
            if third_dim
            else ("x", "y"),
            save_path,
            third_dim
        )

    # =====================================================
    # DEGREE DISTRIBUTION
    # =====================================================

    def degree_distribution(self):

        degrees = [

            len(self.edges[n])

            for n in self.edges
        ]

        plt.figure(figsize=(8, 5))

        plt.hist(
            degrees,
            bins=30
        )

        plt.xlabel("Degree")

        plt.ylabel("Frequency")

        plt.title("IGNG Degree Distribution")

        plt.show()

    # =====================================================
    # CONNECTED COMPONENTS
    # =====================================================

    def connected_components(self):

        G = nx.Graph()

        for i in self.edges:

            for j in self.edges[i]:

                G.add_edge(i, j)

        n = nx.number_connected_components(G)

        print(f"Connected components: {n}")

        return n

    # =====================================================
    # GEXF EXPORT
    # =====================================================

    def to_gexf(
        self,
        path
    ):

        G = nx.Graph()

        for i in range(len(self.nodes)):

            label = (

                int(self.labels[i].argmax())

                if self.labels is not None

                else -1
            )

            attrs = {
                "class_label": label
            }

            # optional coordinates
            if self.nodes.shape[1] >= 2:

                attrs["x"] = float(
                    self.nodes[i, 0]
                )

                attrs["y"] = float(
                    self.nodes[i, 1]
                )

            if self.nodes.shape[1] >= 3:

                attrs["z"] = float(
                    self.nodes[i, 2]
                )

            G.add_node(i, **attrs)

        for i in self.edges:

            for j, age in self.edges[i].items():

                if i < j:

                    G.add_edge(
                        i,
                        j,
                        weight=float(age)
                    )

        nx.write_gexf(G, path)

        print(f"✓ Saved graph: {path}")