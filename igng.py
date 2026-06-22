import torch
import torch.nn as nn

class IGNG(nn.Module):
    """
    Incremental Growing Neural Gas (IGNG)

    Algorithme de clustering incrémental inspiré du Growing Neural Gas (GNG).
    Le réseau construit progressivement une représentation topologique des
    données sous forme d'un graphe dont :

    - les nœuds représentent des prototypes (vecteurs de référence),
    - les arêtes représentent les relations de voisinage entre prototypes,
    - la structure évolue dynamiquement selon les données observées.

    Contrairement aux méthodes de clustering classiques nécessitant un nombre
    de clusters fixé à l'avance, l'IGNG adapte automatiquement sa structure
    pendant l'apprentissage.

    Parameters
    ----------
    input_dim : int
        Dimension des vecteurs d'entrée.

    sigma : float, default=1.0
        Seuil de similarité utilisé pour décider si un nouveau nœud doit
        être créé.

    eps_b : float, default=0.05
        Taux d'apprentissage appliqué au Best Matching Unit (BMU).

    eps_n : float, default=0.005
        Taux d'apprentissage appliqué aux voisins du BMU.

    age_max : int, default=50
        Âge maximal autorisé pour une arête avant sa suppression.

    mature_age : int, default=5
        Nombre minimal de mises à jour nécessaires pour qu'un nœud
        passe de l'état EMBRYO à l'état MATURE.

    max_nodes : int, default=1000
        Nombre maximal de nœuds autorisés dans le réseau.

    device : str, default="cpu"
        Périphérique utilisé pour les calculs ("cpu" ou "cuda").
    """

    EMBRYO = 0
    MATURE = 1

    def __init__(
        self,
        input_dim,
        sigma=1.0,
        eps_b=0.05,
        eps_n=0.005,
        age_max=50,
        mature_age=5,
        max_nodes=1000,
        device="cpu"
    ):
        super().__init__()

        # ==========================================================
        # Hyperparamètres du modèle
        # ==========================================================

        self.input_dim = input_dim
        """
        Dimension de l'espace de représentation des données.
        """

        self.sigma = sigma
        """
        Seuil de distance utilisé pour décider de l'ajout de nouveaux nœuds.
        """

        self.eps_b = eps_b
        """
        Taux d'apprentissage du Best Matching Unit (BMU).
        """

        self.eps_n = eps_n
        """
        Taux d'apprentissage appliqué aux voisins du BMU.
        """

        self.age_max = age_max
        """
        Âge maximal des arêtes avant suppression.
        """

        self.mature_age = mature_age
        """
        Nombre minimal d'activations nécessaires pour qu'un nœud
        soit considéré comme mature.
        """

        self.max_nodes = max_nodes
        """
        Limite supérieure du nombre de nœuds du réseau.
        """

        self.device = device

        # ==========================================================
        # Représentation neuronale
        # ==========================================================

        self.nodes = torch.empty(
            (0, input_dim),
            device=device
        )
        """
        Matrice des prototypes.

        Shape
        -----
        (N, input_dim)

        où N représente le nombre courant de nœuds.
        Chaque ligne correspond au vecteur prototype d'un nœud.
        """

        self.node_age = torch.empty(
            (0,),
            device=device
        )
        """
        Âge de chaque nœud.

        Shape
        -----
        (N,)

        L'âge augmente au fil des itérations et permet notamment
        de distinguer les nœuds récents des nœuds stabilisés.
        """

        self.node_state = torch.empty(
            (0,),
            dtype=torch.long,
            device=device
        )
        """
        État de maturation des nœuds.

        Valeurs possibles
        -----------------
        EMBRYO = 0
            Nœud récemment créé.

        MATURE = 1
            Nœud suffisamment entraîné pour participer
            pleinement à la structure topologique.
        """

        # ==========================================================
        # Structure topologique
        # ==========================================================

        self.edges = {}
        """
        Graphe de voisinage représenté sous forme de liste d'adjacence.

        Format
        ------
        {
            node_i: {
                neighbor_j: edge_age,
                neighbor_k: edge_age
            }
        }

        où :
        - node_i est l'indice du nœud courant,
        - neighbor_j est un voisin connecté,
        - edge_age est l'âge de l'arête.

        Les arêtes trop anciennes sont supprimées afin de maintenir
        une représentation topologique adaptative.
        """

        print(f"IGNG initialized on {device}")

    # =========================================================
    # PUBLIC METHODS
    # =========================================================

    def partial_fit(self, X):
        """
        Met à jour le réseau IGNG à partir d'un ensemble d'échantillons.

        Chaque échantillon est traité séquentiellement afin de permettre
        une construction incrémentale de la structure topologique.

        Pour chaque observation :

        1. Recherche des deux neurones les plus proches.
        2. Vérification du critère de vigilance.
        3. Création éventuelle d'un nouveau neurone.
        4. Adaptation des prototypes existants.
        5. Mise à jour des connexions.
        6. Maturation et élagage du graphe.

        Parameters
        ----------
        X : torch.Tensor
            Ensemble d'échantillons de forme
            (n_samples, input_dim).
        """

        # Déplacement des données vers le périphérique utilisé
        X = X.to(self.device)

        # Traitement séquentiel des échantillons
        for x in X:

            x = x.view(-1)

            # =================================================
            # GRAPHE VIDE
            # =================================================

            # Création du premier neurone
            if self.nodes.shape[0] == 0:

                self._create_neuron(x)

                continue

            # =================================================
            # RECHERCHE DES DEUX BMU
            # =================================================

            bmu, second = self._find_two_closest(x)

            # Distance au meilleur neurone
            d1 = torch.norm(
                x - self.nodes[bmu]
            )

            # =================================================
            # TEST DE VIGILANCE
            # =================================================

            # Échantillon trop éloigné :
            # création d'un nouveau prototype
            if d1 > self.sigma:

                self._create_neuron(x)

                continue

            # =================================================
            # CAS D'UN SEUL NEURONE
            # =================================================

            if second is None:

                new_node = self._create_neuron(x)

                if new_node is None:
                    continue

                # Connexion du nouveau neurone au BMU
                self._connect(
                    bmu,
                    new_node
                )

                continue

            # =================================================
            # TEST DU SECOND BMU
            # =================================================

            d2 = torch.norm(
                x - self.nodes[second]
            )

            # Le second voisin est trop éloigné :
            # création d'un nouveau neurone
            if d2 > self.sigma:

                new_node = self._create_neuron(x)

                if new_node is None:
                    continue

                self._connect(
                    bmu,
                    new_node
                )

                continue

            # =================================================
            # PHASE D'APPRENTISSAGE
            # =================================================

            # Vieillissement des connexions du BMU
            self._increment_edge_ages(bmu)

            # Adaptation des prototypes
            self._adapt(
                bmu,
                x
            )

            # Connexion entre les deux BMU
            self._connect(
                bmu,
                second
            )

            # Mise à jour de l'âge des voisins
            self._increment_neighbor_ages(bmu)

            # Passage éventuel à l'état mature
            self._mature_nodes()

            # Suppression des éléments obsolètes
            self._prune()

    def predict(self, X):
        """
        Associe chaque échantillon au neurone le plus proche.

        Cette méthode effectue une quantification vectorielle
        en retournant l'indice du Best Matching Unit (BMU)
        pour chaque observation.

        Parameters
        ----------
        X : torch.Tensor
            Ensemble d'échantillons de forme
            (n_samples, input_dim).

        Returns
        -------
        list[int]
            Liste des indices des BMU associés
            à chaque échantillon.

            La valeur -1 est retournée si le réseau
            ne contient aucun neurone.
        """

        # Déplacement des données vers le périphérique utilisé
        X = X.to(self.device)

        predictions = []

        # Recherche du BMU pour chaque échantillon
        for x in X:

            x = x.view(-1)

            # Réseau vide
            if self.nodes.shape[0] == 0:

                predictions.append(-1)

                continue

            # Recherche du neurone le plus proche
            bmu = self._find_bmu(x)

            predictions.append(bmu)

        return predictions        

    # =========================================================
    # PRIVATE METHODS
    # =========================================================

    def _create_neuron(self, x):
        """
        Crée un nouveau neurone dans le réseau.

        Cette méthode ajoute un prototype correspondant au vecteur d'entrée
        fourni. Le nouveau neurone est initialisé dans l'état EMBRYO,
        avec un âge nul et sans connexion à d'autres neurones.

        Parameters
        ----------
        x : torch.Tensor
            Vecteur de données utilisé pour initialiser le prototype
            du nouveau neurone.

            Shape
            -----
            (input_dim,)

        Returns
        -------
        int or None
            Identifiant du neurone créé.

            - Retourne l'indice du nouveau neurone si la création
            a réussi.
            - Retourne None si le nombre maximal de neurones
            (`max_nodes`) a été atteint.

        Notes
        -----
        Lors de la création :

        1. Le vecteur prototype est ajouté à `self.nodes`.
        2. L'âge du neurone est initialisé à 0.
        3. Son état est défini à EMBRYO.
        4. Une entrée vide est créée dans la liste d'adjacence
        afin de permettre l'ajout futur de connexions.

        Examples
        --------
        >>> x = torch.tensor([0.2, 0.8])
        >>> neuron_id = igng._create_neuron(x)
        >>> print(neuron_id)
        0
        """
        if self.nodes.shape[0] >= self.max_nodes:
            return None

        new_id = self.nodes.shape[0]

        # add node
        self.nodes = torch.cat([
            self.nodes,
            x.unsqueeze(0)
        ])

        # age = 0
        self.node_age = torch.cat([
            self.node_age,
            torch.tensor([0.0], device=self.device)
        ])

        # embryo
        self.node_state = torch.cat([
            self.node_state,
            torch.tensor([self.EMBRYO], device=self.device)
        ])

        self.edges[new_id] = {}

        return new_id

    def _find_bmu(self, x):
        """
        Find the Best Matching Unit (BMU).

        Computes the Euclidean distance between the input sample and
        all neuron prototypes, then returns the index of the closest neuron.

        Parameters
        ----------
        x : torch.Tensor
            Input sample of shape (input_dim,).

        Returns
        -------
        int
            Index of the nearest neuron.
        """
        distances = torch.norm(
            self.nodes - x,
            dim=1
        )

        return torch.argmin(distances).item()

    def _find_two_closest(self, x):
        """
        Recherche les deux neurones les plus proches d'un échantillon.

        Calcule la distance euclidienne entre x et tous les prototypes,
        puis retourne les indices des deux neurones les plus proches.

        Parameters
        ----------
        x : torch.Tensor
            Échantillon d'entrée.

        Returns
        -------
        tuple
            (bmu1, bmu2) correspondant aux deux neurones les plus proches.
            Si un seul neurone existe, retourne (0, None).
        """

        if self.nodes.shape[0] == 1:
            return 0, None

        # Distances entre x et tous les prototypes
        distances = torch.norm(
            self.nodes - x,
            dim=1
        )

        # Sélection des deux plus petites distances
        idx = torch.topk(
            distances,
            2,
            largest=False
        )[1]

        return idx[0].item(), idx[1].item()

    def _adapt(self, bmu, x):
        """
        Adapte les prototypes du BMU et de ses voisins.

        Le BMU est déplacé vers l'échantillon x avec un taux
        d'apprentissage élevé, tandis que ses voisins sont
        déplacés avec un taux plus faible.

        Parameters
        ----------
        bmu : int
            Indice du Best Matching Unit.

        x : torch.Tensor
            Échantillon d'entrée.
        """

        # Mise à jour du BMU
        self.nodes[bmu] += (
            self.eps_b *
            (x - self.nodes[bmu])
        )

        # Mise à jour des voisins
        neighbors = list(self.edges[bmu].keys())

        for n in neighbors:
            self.nodes[n] += (
                self.eps_n *
                (x - self.nodes[n])
            )

    def _connect(self, a, b):
        """
        Crée ou renouvelle une connexion entre deux neurones.

        La connexion est bidirectionnelle et son âge est
        réinitialisé à zéro.

        Parameters
        ----------
        a : int
            Premier neurone.

        b : int
            Second neurone.
        """

        if a == b:
            return

        if a not in self.edges:
            self.edges[a] = {}

        if b not in self.edges:
            self.edges[b] = {}

        # Réinitialisation de l'âge de l'arête
        self.edges[a][b] = 0
        self.edges[b][a] = 0

    def _increment_edge_ages(self, node):
        """
        Incrémente l'âge de toutes les arêtes incidentes
        à un neurone donné.

        Parameters
        ----------
        node : int
            Indice du neurone concerné.
        """

        neighbors = list(self.edges[node].keys())

        for n in neighbors:
            self.edges[node][n] += 1
            self.edges[n][node] += 1

    def _increment_neighbor_ages(self, bmu):
        """
        Incrémente l'âge des voisins du BMU.

        Cette opération permet de suivre l'ancienneté des
        neurones connectés au BMU.

        Parameters
        ----------
        bmu : int
            Indice du Best Matching Unit.
        """

        neighbors = list(self.edges[bmu].keys())
        self.node_age[bmu] += 1
        for n in neighbors:
            self.node_age[n] += 1

    def _mature_nodes(self):
        """
        Met à jour l'état des neurones.

        Tous les neurones dont l'âge atteint ou dépasse
        le seuil `mature_age` passent de l'état EMBRYO
        à l'état MATURE.
        """

        # Sélection des neurones suffisamment âgés
        mature_mask = (
            self.node_age >= self.mature_age
        )

        # Changement d'état
        self.node_state[mature_mask] = self.MATURE

    def _prune(self):
        """
        Nettoie la structure du réseau.

        Cette méthode réalise deux opérations :

        1. Supprime les arêtes dont l'âge dépasse `age_max`.
        2. Supprime les neurones isolés encore à l'état EMBRYO.

        Après suppression, les tenseurs contenant les prototypes,
        les âges et les états sont reconstruits afin de conserver
        une indexation cohérente des neurones. Le graphe de voisinage
        est ensuite réindexé en conséquence.

        Notes
        -----
        Les neurones matures ne sont jamais supprimés uniquement
        parce qu'ils sont isolés. Cette règle permet de préserver
        les connaissances déjà acquises par le réseau.
        """

        # =====================================================
        # SUPPRESSION DES ARÊTES TROP ANCIENNES
        # =====================================================

        for node in list(self.edges.keys()):

            # Vérifie que le nœud existe toujours
            if node not in self.edges:
                continue

            neighbors = list(self.edges[node].keys())

            for n in neighbors:

                # Vérifie que le voisin existe toujours
                if n not in self.edges:
                    continue

                # Supprime les connexions dont l'âge dépasse le seuil
                if self.edges[node][n] > self.age_max:

                    # Suppression dans les deux directions
                    self.edges[node].pop(n, None)
                    self.edges[n].pop(node, None)

        # =====================================================
        # RECHERCHE DES EMBRYONS ISOLÉS
        # =====================================================

        to_remove = []

        for node in range(self.nodes.shape[0]):

            if node not in self.edges:
                continue

            # Un embryon sans voisin est considéré comme inutile
            if (
                len(self.edges[node]) == 0
                and self.node_state[node] == self.EMBRYO
            ):
                to_remove.append(node)

        # Aucun neurone à supprimer
        if len(to_remove) == 0:
            return

        # =====================================================
        # DÉTERMINATION DES NEURONES CONSERVÉS
        # =====================================================

        keep = [
            i for i in range(self.nodes.shape[0])
            if i not in to_remove
        ]

        # Correspondance anciens indices → nouveaux indices
        old_to_new = {
            old: new
            for new, old in enumerate(keep)
        }

        # =====================================================
        # RECONSTRUCTION DES TENSEURS
        # =====================================================

        # Conservation uniquement des neurones restants
        self.nodes = self.nodes[keep]

        self.node_age = self.node_age[keep]

        self.node_state = self.node_state[keep]

        # =====================================================
        # RECONSTRUCTION DU GRAPHE
        # =====================================================

        new_edges = {}

        for old_node in keep:

            new_node = old_to_new[old_node]

            new_edges[new_node] = {}

            if old_node not in self.edges:
                continue

            for old_neighbor, age in self.edges[old_node].items():

                # Ignore les voisins supprimés
                if old_neighbor not in old_to_new:
                    continue

                new_neighbor = old_to_new[old_neighbor]

                # Recrée l'arête avec les nouveaux indices
                new_edges[new_node][new_neighbor] = age

        # Remplace l'ancien graphe par le nouveau
        self.edges = new_edges

    # =========================================================
    # UTILITIES
    # =========================================================

    def save(self, path):
        """
        Sauvegarde l'état complet du réseau IGNG.

        Les prototypes, les âges, les états des neurones
        ainsi que la structure du graphe sont enregistrés
        dans un fichier PyTorch.

        Parameters
        ----------
        path : str
            Chemin du fichier de sauvegarde.
        """

        # Sauvegarde de tous les éléments du modèle
        torch.save({
            "nodes": self.nodes,
            "node_age": self.node_age,
            "node_state": self.node_state,
            "edges": self.edges
        }, path)


    def load(self, path):
        """
        Charge un modèle IGNG précédemment sauvegardé.

        Restaure les prototypes, les âges, les états des
        neurones et la structure du graphe.

        Parameters
        ----------
        path : str
            Chemin du fichier de sauvegarde.
        """

        # Chargement du fichier sur le périphérique courant
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        # Restauration des prototypes
        self.nodes = checkpoint["nodes"]

        # Restauration des âges des neurones
        self.node_age = checkpoint["node_age"]

        # Restauration des états EMBRYO/MATURE
        self.node_state = checkpoint["node_state"]

        # Restauration de la structure du graphe
        self.edges = checkpoint["edges"]

        print("IGNG loaded successfully.")