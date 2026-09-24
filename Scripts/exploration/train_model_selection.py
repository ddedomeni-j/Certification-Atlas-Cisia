from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_score)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_preparation import build_preprocessor_scenario1, build_preprocessor_scenario3, load_dataset_scenario1

# gestion warning lié à la normalisation des tokens
import warnings
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent with your preprocessing")

DATASET_NAME = "dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv"
DATASET_SOURCE = "./data/"
DATASET_VERSION = "2026-07-22"

models = {"dummy": DummyClassifier(strategy="most_frequent"),
          "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
          "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
          "xgboost":XGBClassifier(objective="multi:softprob",
                                  num_class=3,
                                  random_state=42,
                                  eval_metric="mlogloss"),
          "lightgbm": LGBMClassifier(objective="multiclass",
                                     num_class=3,
                                     random_state=42,
                                     verbose=-1)}

X, y = load_dataset_scenario1(DATASET_SOURCE+DATASET_NAME)
scenario = 3
for name, model in models.items():
    # Construction du pipeline ML
    if scenario == 1 or scenario == 2:
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor_scenario1()),
                                   ("model", model)])
    elif scenario == 3:
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor_scenario3()),
                                   ("model", model)])
    
    # Cross Validation 5 folds préservant la répartition
    cv = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=42)

    scores = cross_val_score(pipeline,
                             X,
                             y,
                             cv=cv,
                             scoring="f1_macro")

    print(f"{name}: "
          f"{scores.mean():.3f} "
          f"+/- {scores.std():.3f}")
    print("\n---------------\n")


# Conclusion données tabulaires :
# ===============================

# dummy: 0.205 +/- 0.000
# ---------------
# logistic_regression: 0.544 +/- 0.032
# ---------------
# random_forest: 0.606 +/- 0.029
# ---------------
# hist_gradient_boosting: 0.591 +/- 0.018

# La différence de scrore entre le modèle Dummy et les 3 modèles de ML montre que les features contiennent bien de l'information prédictive permettant d'entraîner un modèle
# Random Forest a le meilleur score, hist_gradient_boosting étant relativement proche. Le faible nombre d'observation ne permet pas de conclure que RandomForest est un meilleur choix que HistGradientBoosting
# HistGradientBoosting est plus stable que RandomForest

# Conclusion multimodal (avec variables sensibles)
# ================================================
# dummy: 0.205 +/- 0.000
# ---------------
# logistic_regression: 0.669 +/- 0.020
# ---------------
# random_forest: 0.668 +/- 0.018
# ---------------
# xgboost: 0.693 +/- 0.012
# ---------------
# lightgbm: 0.697 +/- 0.015

# La différence de scrore entre le modèle Dummy et les 3 modèles de ML montre que les features contiennent bien de l'information prédictive permettant d'entraîner un modèle
# Le premier scénario combine l'ensemble des informations disponibles : variables tabulaires structurées 
# et synthèse textuelle issue de l'entretien d'aiguillage. Les variables à forte cardinalité ont fait l'objet 
# d'une ingénierie de caractéristiques avec l'extraction du département à partir du code INSEE. Les données textuelles 
# ont été vectorisées par TF-IDF avant l'apprentissage.

# Les résultats montrent un gain important par rapport aux modèles entraînés uniquement sur les variables tabulaires. 
# Le meilleur score est obtenu par LightGBM avec un F1 macro moyen de 0,697 ± 0,015, suivi de XGBoost (0,693 ± 0,012). 
# Ces résultats suggèrent que la synthèse d'entretien contient une information prédictive déterminante pour l'estimation 
# du délai de retour à l'emploi.

# On garde pour l'analyse détaillée :
# logistic_regression qui combine bon résultat et explicabilité forte
# XGBoost bonne qualité, plus stable que lightGBM, radicalement différent de logistic regression

# Apres suppression stopwords

# dummy: 0.205 +/- 0.000
# ---------------
# logistic_regression: 0.670 +/- 0.021
# ---------------
# random_forest: 0.670 +/- 0.024
# ---------------
# xgboost: 0.684 +/- 0.021
# ---------------
# lightgbm: 0.693 +/- 0.023
# ---------------

# Analyse scénario 2 : sans la variable nationalite hors ue
# =========================================================

# dummy: 0.205 +/- 0.000
# ---------------
# logistic_regression: 0.657 +/- 0.019
# ---------------
# random_forest: 0.667 +/- 0.019
# ---------------
# xgboost: 0.667 +/- 0.020
# ---------------
# lightgbm: 0.666 +/- 0.013
# ---------------

# 1. Random Forest ≈ XGBoost ≈ LightGBM
# 2. Logistic Regression
# La variable sensible contribue effectivement aux performances du système.
# Sa suppression entraîne une dégradation mesurable des performances prédictives, particulièrement pour les modèles de gradient boosting.
# Cette observation suggère que la variable contient une information statistiquement corrélée au délai de retour à l'emploi dans le jeu de données étudié.

# Sans age et nationalite

# dummy: 0.205 +/- 0.000
# ---------------
# logistic_regression: 0.640 +/- 0.025
# ---------------
# random_forest: 0.631 +/- 0.013
# ---------------
# xgboost: 0.600 +/- 0.029
# ---------------
# lightgbm: 0.594 +/- 0.027
# ---------------

# Conclusion scenario 3
# =====================
# dummy: 0.205 +/- 0.000
# ---------------
# logistic_regression: 0.635 +/- 0.008
# ---------------
# random_forest: 0.635 +/- 0.008
# ---------------
# xgboost: 0.634 +/- 0.007
# ---------------
# lightgbm: 0.634 +/- 0.007
# ---------------

# ### Scénario 3 : Prédiction à partir du texte seul

# Le scénario 3 vise à évaluer la capacité du système à prédire le délai de retour à l'emploi à partir de la seule synthèse 
# rédigée par le conseiller.

# Les résultats obtenus montrent que les performances restent relativement élevées malgré la suppression de l'ensemble 
# des variables structurées. Le meilleur modèle obtient un F1 macro de 0.635, soit près de 90 % des performances du scénario 
# complet (0.711).

# Cette observation montre que les verbatims contiennent déjà une part importante de l'information utile à la prédiction. 
# L'analyse SHAP confirme que les termes liés aux difficultés rencontrées, au projet professionnel, aux compétences ou aux freins
# périphériques influencent fortement les décisions du modèle.

# Toutefois, les performances restent inférieures à celles observées dans le scénario multimodal et le nombre d'erreurs critiques
# augmente sensiblement, en particulier pour LightGBM. Les données structurées conservent donc une valeur ajoutée
# importante pour sécuriser les décisions les plus sensibles.

# Le scénario 3 démontre ainsi la richesse informationnelle des synthèses d'entretien mais confirme également l'intérêt
# d'une approche multimodale combinant texte et données structurées.

