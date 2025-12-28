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
The goal of this project is to propose a **method to analyze and evaluate the spaces controlled by an individual player in well-defined game situations**.

My initial observation is that most existing studies leveraging Pitch Control models focus on space occupation at the team level, rather than at the individual player level. However, analyzing space control per player makes it possible to address several important questions:
- How does a player position himself relative to other players (which is precisely the advantage of Pitch Control over simple positional heatmaps)?
- What types of spaces does a player usually control? What is their space control profile?
- Are these controlled spaces relevant (i.e. reachable by a pass) and dangerous (i.e. do they increase the probability of scoring)?

This approach is especially relevant for positions that interact less with the ball, such as forwards. It enables a more holistic evaluation of player performance: a player who rarely touches the ball and produces few decisive actions does not necessarily have a poor game. They may still control accessible and dangerous spaces without being served, which in turn could highlight limitations in their teammates’ decision-making.

#### Methods
The methodology is structured into three main steps (each time filtering on a specific player_position and game_situation):

**Individual Pitch Control (IPV)**
We compute Pitch Control at the individual level by adapting [William Spearman’s physics-based Pitch Control model](https://www.researchgate.net/publication/334849056_Quantifying_Pitch_Control).
Instead of summing the control probabilities of all players from both teams, we compute the probability of a single player against the sum of the probabilities of the opposing team’s players.

**Spatial categorization of controlled areas**
To better interpret the controlled spaces, we cluster them using unsupervised learning methods (a K-means model produces coherent spatial zones, and a UMAP-HDBSCAN model allows for a finer characterization of space shapes)

**Evaluation of controlled spaces**
To assess the value of these spaces, we train xPass and xT models based on the [deep-learning Soccermap approach](https://arxiv.org/abs/2010.10202).

We introduce the metric **iscT-Δ — Individual Space Controlled Threat Delta**:
- It is defined as the weighted average of the difference between the xT values of the spaces controlled by the player and the xT of where the ball is, weighted by both the Individual Pitch Control and the xPass probability.
- Intuition: the more likely a player is to control a space — and the more likely a pass into that space is to be successful — the more weight is assigned to the incremental xT gain.
<img src="images/isct_delta.png" width="600">

#### Results
The results highlight several key findings:
- Players exhibit very diverse profiles in terms of the types of spaces they control.
- Truly “complete” players are rare: no player consistently creates relevant and dangerous spaces (iscT-Δ) across all game situations; instead, each player tends to have preferred contexts in which they are most effective.
- Use case example: for a given game situation, one can focus on the three best and three worst frames according to the iscT-Δ metric in order to provide concrete positional recommendations to a player.

#### Conclusion
This method is computationally demanding, as it strongly depends on the quality of the Soccermap models (which themselves require a large number of frames). However, it provides a more holistic view of a player’s true performance.

As a next step, a promising direction would be to integrate this framework into an EPV-based model, enabling a unified valuation of space, actions, and outcomes.