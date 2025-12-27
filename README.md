# SkillCorner X PySport Analytics Cup
This repository contains the submission template for the SkillCorner X PySport Analytics Cup **Research Track**. 
Your submission for the **Research Track** should be on the `main` branch of your own fork of this repository.

Find the Analytics Cup [**dataset**](https://github.com/SkillCorner/opendata/tree/master/data) and [**tutorials**](https://github.com/SkillCorner/opendata/tree/master/resources) on the [**SkillCorner Open Data Repository**](https://github.com/SkillCorner/opendata).

## Submitting
Make sure your `main` branch contains:
1. A single Jupyter Notebook in the root of this repository called `submission.ipynb`
    - This Juypter Notebook can not contain more than 2000 words.
    - All other code should also be contained in this repository, but should be imported into the notebook from the `src` folder.
2. An abstract of maximum 500 words that follows the **Research Track Abstract Template**.
    - The abstract can contain a maximum of 2 figures, 2 tables or 1 figure and 1 table.
3. Submit your GitHub repository on the [Analytics Cup Pretalx page](https://pretalx.pysport.org)

Finally:
- Make sure your GitHub repository does **not** contain big data files. The tracking data should be loaded directly from the [Analytics Cup Data GitHub Repository](https://github.com/SkillCorner/opendata).For more information on how to load the data directly from GitHub please see this [Jupyter Notebook](https://github.com/SkillCorner/opendata/blob/master/resources/getting-started-skc-tracking-kloppy.ipynb).
- Make sure the `submission.ipynb` notebook runs on a clean environment.

_⚠️ Not adhering to these submission rules and the [**Analytics Cup Rules**](https://pysport.org/analytics-cup/rules) may result in a point deduction or disqualification._

---

## Research Track Abstract Template (max. 500 words)
#### Introduction
L'objectif de ce projet est de créer une méthode qui permettrait d'analyser et d'évaluer les espaces contrôlés par un joueur au cours de situation de jeu bien précises.
Mon constat de base : les principales études qui exploitent les modèles de Pitch Control se concentrent sur l'occupation des espaces par une équipe dans son ensemble, et pas spécifiquement par un joueur.
Pourtant, cela permet de répondre à de nombreuses questions :
- Comment un joueur se positionne par rapport aux autres joueurs (c'est ça l'intérêt d'analyser le Pitch Control plutôt qu'une heatmap de positions de jeu) ?
- Quels sont les types d'espaces qu'il a pour habitude de contrôler ? Quel est son profil de contrôle d'espace ?
- Ces espaces contrôlés sont-ils pertinents (accessibles par la passe) et dangereux (augmentent la proba de marquer) ?

Cette approche est encore plus pertinente pour les postes qui ont moins le ballon, comme les attaquants par exemple.
Elle permet d'ailleurs d'apporter une évaluation plus globale d'un joueur : un joueur qui touche peu le ballon et fait peu d'actions décisives fait-il forcément un mauvais match ? Pas forcément : il peut potentiellement contrôlés des espaces accessibles et dangereux sans jamais être servis, ce qui dans ce cas pourrait également permettre de pénaliser la performance de ses coéquipiers.

#### Methods
La méthode se décompose en 3 étapes principales (à chaque fois en filtrant sur une player_position et une game_situation précises):
- On calcule le Pitch Control au niveau individuel ("IPV" : Individual Pitch Control) : on applique [la méthode de Pitch Control physics-based de William Spearman](https://www.researchgate.net/publication/334849056_Quantifying_Pitch_Control), sauf qu'au lieu de sommer les probas de tous les joueurs des 2 équipes, on prend la proba d'un seul joueur contre la somme des probas des joueurs adverses.
- On décide ensuite de catégoriser ces espaces pour mieux les interpréter, à l'aide d'une méthode non supervisée. Un modèle K-means donne des zones très cohérentes, et un modèle Hdbscan (précédé d'une réduction de dimension non linéaire comme Umap) permet d'aller en plus loin en distinguant les format des espaces (mais c'est beaucoup plus long pour fitter).
- Enfin, afin d'évaluer ces espaces, on entraîne des modèles de xPass et xT à partir de [la méthode de Deep Learning CNN Soccermap](https://arxiv.org/abs/2010.10202).

We introduce the metric **"iscT" : Individual Space Controlled Threat** :
- It is the weighted average of the xT of the spaces controlled by the player, weighted by the Individual Pitch Control and the xPass probabilities
- Intuition : the more likely the player is to control the space — and for a pass into that space to be successful — the more weight we assign to the xT


To make this metric usable — i.e. to be able to compare different situations and different players — we prefer to analyze how much these controlled spaces would increase the probability of scoring compared to the on-ball xT : **iscT-Δ : Individual Space Controlled Threat Increase**

#### Results
La promesse de ce repo : il suffit de donner en input une liste de macth_id, une player_position et une game_situation, et le code effectura toutes les étapes listées au-dessus pour tous ces matchs.

#### Conclusion
