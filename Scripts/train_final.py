from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from data_preparation import build_preprocessor_scenario1, build_preprocessor_scenario3, load_dataset_scenario1

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import shap

import time

# gestion warning lié à la normalisation des tokens
import warnings
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent with your preprocessing")

DATASET_NAME = "dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv"
DATASET_SOURCE = "./data/"
DATASET_VERSION = "2026-07-22"

models = {"random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
          "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42)}

X, y = load_dataset_scenario1(DATASET_SOURCE+DATASET_NAME)

X_trainval, X_unseen, y_trainval, y_unseen = train_test_split(X,
                                                              y,
                                                              test_size=0.20,
                                                              stratify=y,
                                                              random_state=42)

cv = StratifiedKFold(n_splits=5,
                     shuffle=True,
                     random_state=42)

# # Test scénario tabulaire
# # =======================
# # Random forest GridSearch
# rf_params = {"model__n_estimators": [100, 300, 500],
#              "model__max_depth": [None, 10, 20],
#              "model__min_samples_leaf": [1, 5, 10]}

# rf_pipeline = Pipeline([("prep", build_preprocessor()),
#                         ("model",RandomForestClassifier(class_weight="balanced", random_state=42))])

# rf_search = GridSearchCV(estimator=rf_pipeline,
#                          param_grid=rf_params,  
#                          scoring="f1_macro",
#                          cv=cv,
#                          n_jobs=-1)

# rf_search.fit(X_trainval, y_trainval)

# # Hist gradient boost GridSearch
# hgb_pipeline = Pipeline([("prep", build_preprocessor()),
#                          ("model", HistGradientBoostingClassifier(random_state=42))])

# hgb_params = {"model__learning_rate": [0.01, 0.05, 0.1],
#               "model__max_depth": [None, 5, 10],
#               "model__max_iter": [100, 300]}

# hgb_search = GridSearchCV(estimator=hgb_pipeline,
#                           param_grid=hgb_params,
#                           scoring="f1_macro",
#                           cv=cv,
#                           n_jobs=-1)

# hgb_search.fit(X_trainval, y_trainval)

# print("\n---------------\n")
# print(rf_search.best_score_)
# print(rf_search.best_params_)

# print("\n---------------\n")
# print(hgb_search.best_score_)
# print(hgb_search.best_params_)

# Conclusion scénario tabulaire

# Une phase d'optimisation des hyperparamètres a été réalisée par validation croisée stratifiée à 5 folds 
# sur les deux modèles les plus performants issus du benchmark initial. 
# Le meilleur résultat est obtenu avec HistGradientBoostingClassifier (F1 macro = 0.629) qui semble avoir une marge 
# de progression supérieure au RandomForestClassifier (F1 macro = 0.605). 
# Le modèle HistGradientBoosting est retenu pour l'évaluation finale.

# Test scénario 1
# ===============

# Logistic regression
lr_params = {"model__C": [0.01, 0.1, 1, 10],
             "model__solver": ["lbfgs"]}

scenario = 3
if scenario == 1 or scenario == 2:
    lr_pipeline = Pipeline([("prep", build_preprocessor_scenario1()),
                            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
elif scenario == 3:
    lr_pipeline = Pipeline([("prep", build_preprocessor_scenario3()),
                            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    
lr_search = GridSearchCV(estimator=lr_pipeline,
                         param_grid=lr_params,  
                         scoring="f1_macro",
                         cv=cv,
                         n_jobs=-1)

start = time.perf_counter()
lr_search.fit(X_trainval, y_trainval)
train_time_lr = time.perf_counter() - start

best_pipeline = lr_search.best_estimator_
start = time.perf_counter()
y_pred = best_pipeline.predict(X_unseen)
inference_time_lr = time.perf_counter() - start
critical_errors_lr = np.sum((y_unseen == 2) &(y_pred == 0))
n_true_2 = np.sum(y_unseen == 2)
critical_rate_lr = (critical_errors_lr / n_true_2)

# Analyse de la feature nationalité
# =================================

best_pipeline_lr = lr_search.best_estimator_
lr_model = best_pipeline_lr.named_steps["model"]
coef = lr_model.coef_
preprocessor = best_pipeline_lr.named_steps["prep"]
feature_names_lr = preprocessor.get_feature_names_out()
if (len(np.where(feature_names_lr == "cat__nationalite_hors_ue_1")[0]) != 0):
    nationality_idx = np.where(feature_names_lr == "cat__nationalite_hors_ue_1")[0][0]

    print("\nInfluence nationalité sur les prédictions du modèle de régression")
    print(coef[:, nationality_idx])
    print("\n")

feature_importance = pd.DataFrame({"feature": feature_names_lr,"coef": lr_model.coef_[2]})
feature_importance["abs_coef"] = (feature_importance["coef"].abs())
feature_importance = (feature_importance.sort_values("abs_coef", ascending=False))
print(feature_importance.head(20))
print("\n")

# # XGBoost
# xgb_params = {"model__n_estimators": [100, 300],
#               "model__max_depth": [3, 5, 7],
#               "model__learning_rate": [0.01, 0.05, 0.1]}

# xgb_pipeline = Pipeline([("prep", build_preprocessor_scenario1()),
#                          ("model", XGBClassifier(objective="multi:softprob", num_class=3, random_state=42, eval_metric="mlogloss"))])

# xgb_search = GridSearchCV(estimator=xgb_pipeline,
#                           param_grid=xgb_params,  
#                           scoring="f1_macro",
#                           cv=cv,
#                           n_jobs=-1)

# start = time.perf_counter()
# xgb_search.fit(X_trainval, y_trainval)
# train_time_xgb = time.perf_counter() - start

# best_pipeline = xgb_search.best_estimator_
# start = time.perf_counter()
# y_pred = best_pipeline.predict(X_unseen)
# inference_time_xgb = time.perf_counter() - start
# critical_errors_xgb = np.sum((y_unseen == 2) & (y_pred == 0))
# n_true_2 = np.sum(y_unseen == 2)
# critical_rate_xgb = (critical_errors_xgb / n_true_2)

# LightGBM
lgbm_params = {"model__n_estimators": [100, 300, 500],
               "model__max_depth": [3, 5, 7],
               "model__learning_rate": [0.01, 0.05, 0.1]}


if scenario == 1 or scenario == 2:
    lgbm_pipeline = Pipeline([("prep", build_preprocessor_scenario1()),
                              ("model", LGBMClassifier(objective="multiclass", num_class=3, random_state=42, verbose=-1))])
elif scenario == 3:
    lgbm_pipeline = Pipeline([("prep", build_preprocessor_scenario3()),
                              ("model", LGBMClassifier(objective="multiclass", num_class=3, random_state=42, verbose=-1))])

lgbm_search = GridSearchCV(estimator=lgbm_pipeline,
                           param_grid=lgbm_params,  
                           scoring="f1_macro",
                           cv=cv,
                           n_jobs=-1)

start = time.perf_counter()
lgbm_search.fit(X_trainval, y_trainval)
train_time_lgbm = time.perf_counter() - start

best_pipeline_lgbm = lgbm_search.best_estimator_
start = time.perf_counter()
y_pred = best_pipeline_lgbm.predict(X_unseen)
inference_time_lgbm = time.perf_counter() - start
critical_errors_lgbm = np.sum((y_unseen == 2) &(y_pred == 0))
n_true_2 = np.sum(y_unseen == 2)
critical_rate_lgbm = (critical_errors_lgbm / n_true_2)

# Trouver une bonne prédiction
correct_idx = np.where(y_pred == y_unseen)[0]
print(f"{len(correct_idx)} bonnes prédictions")
good_idx = correct_idx[0]

# Trouver une mauvaise prédiction
critical_idx = np.where((y_unseen == 2) & (y_pred == 0))[0]
print(f"{len(critical_idx)} mauvaises prédictions")
bad_idx = critical_idx[0]

# SHAP
X_train_transformed = best_pipeline_lgbm.named_steps["prep"].transform(X_unseen)
model = best_pipeline_lgbm.named_steps["model"]
preprocessor = best_pipeline_lgbm.named_steps["prep"]
feature_names = preprocessor.get_feature_names_out()
explainer = shap.TreeExplainer(model)
X_train_dense = X_train_transformed.toarray()
shap_values = explainer(X_train_dense)
shap_values.feature_names = feature_names

plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_values[good_idx, :, 2])
plt.title("Bonne prédiction")
plt.tight_layout()

plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_values[bad_idx, :, 2])
plt.title("Erreur critique 2 → 0")
plt.tight_layout()

# good_case = X_unseen.iloc[[good_idx]].T
# good_case.columns = ["Bonne prédiction"]
# bad_case = X_unseen.iloc[[bad_idx]].T
# bad_case.columns = ["Erreur critique"]

# comparison = pd.concat([good_case, bad_case], axis=1)
# print(comparison)

idx = good_idx
contrib = pd.DataFrame({"feature": feature_names, 
                        "value": X_train_transformed[idx].toarray().ravel(),
                        "shap": shap_values.values[idx, :, 2]})
contrib["abs_shap"] = contrib["shap"].abs()
print(contrib.sort_values("abs_shap", ascending=False).head(10))
print("\n---------------\n")
idx = bad_idx
contrib = pd.DataFrame({"feature": feature_names, 
                        "value": X_train_transformed[idx].toarray().ravel(),
                        "shap": shap_values.values[idx, :, 2]})
contrib["abs_shap"] = contrib["shap"].abs()
print(contrib.sort_values("abs_shap", ascending=False).head(10))

# importance globale
plt.figure(figsize=(8, 6))
shap.plots.bar(shap_values[:, :, 2], max_display=15)
plt.tight_layout()

# Poids moyen de la feauture nationalité pour lightGBM
if (len(np.where(feature_names_lr == "cat__nationalite_hors_ue_1")[0]) != 0):
    feature_idx = np.where(feature_names == "cat__nationalite_hors_ue_1")[0][0]
    mean_abs_shap = np.mean(np.abs(shap_values.values[:, feature_idx, 2]))
    print(f"Valeur moyenne du poids de la feature cat__nationalite_hors_ue_1 : {mean_abs_shap}")

    idx_hors_ue = X_unseen["nationalite_hors_ue"] == 1
    idx_ue = X_unseen["nationalite_hors_ue"] == 0
    print(f"Valeur moyenne du poids de la feature cat__nationalite_hors_ue_1 = 1 : {shap_values.values[idx_hors_ue, feature_idx, 2].mean()}")
    print(f"Valeur moyenne du poids de la feature cat__nationalite_hors_ue_1 = 0 : {shap_values.values[idx_ue, feature_idx, 2].mean()}")

# Resultats par modele
# ====================

print("\n---------------\n")
print("LogisticRegression")
print(f"Meilleur score : {lr_search.best_score_}")
print(f"Meilleur set de paramètres : {lr_search.best_params_}")
print(f"Durée de l'entrainement : {train_time_lr:.3f} s")
print(f"Temps par observation : {1000 * inference_time_lr / len(X_unseen):.4f} ms")
print(f"Erreurs critiques (2 -> 0) : {critical_errors_lr}")
print(f"Taux d'erreurs critiques (2 -> 0) : {critical_rate_lr}")

# print("\n---------------\n")
# print("XGBoost")
# print(xgb_search.best_score_)
# print(xgb_search.best_params_)
# print(f"{train_time_xgb:.3f} s")
# print(f"Temps par observation : {1000 * inference_time_xgb / len(X_unseen):.4f} ms")
# print(f"Erreurs critiques (2 -> 0) : {critical_errors_xgb}")
# print(f"Taux d'erreurs critiques (2 -> 0) : {critical_rate_xgb}")

print("\n---------------\n")
print("lightGBM")
print(f"Meilleur score : {lgbm_search.best_score_}")
print(f"Meilleur set de paramètres : {lgbm_search.best_params_}")
print(f"Durée de l'entrainement : {train_time_lgbm:.3f} s")
print(f"Temps par observation : {1000 * inference_time_lgbm / len(X_unseen):.4f} ms")
print(f"Erreurs critiques (2 -> 0) : {critical_errors_lgbm}")
print(f"Taux d'erreurs critiques (2 -> 0) : {critical_rate_lgbm}")

# Explorer les erreurs
# ====================
y_pred = best_pipeline_lgbm.predict(X_unseen)
y_proba = best_pipeline_lgbm.predict_proba(X_unseen)

critical_mask = ((y_unseen == 2) & (y_pred == 0))
critical_errors = X_unseen[critical_mask].copy()
critical_errors["classe_reelle"] = 2
critical_errors["classe_predite"] = 0
critical_errors["score_pred"] = y_proba[critical_mask].max(axis=1)

pd.set_option("display.max_columns", None)
print("\n Exploration des cas d'erreur lightGBM :")
print(critical_errors)

y_pred = best_pipeline_lr.predict(X_unseen)
y_proba = best_pipeline_lr.predict_proba(X_unseen)

critical_mask = ((y_unseen == 2) & (y_pred == 0))
critical_errors = X_unseen[critical_mask].copy()
critical_errors["classe_reelle"] = 2
critical_errors["classe_predite"] = 0
critical_errors["score_pred"] = y_proba[critical_mask].max(axis=1)

pd.set_option("display.max_columns", None)
print("\n Exploration des cas d'erreur regression logistique :")
print(critical_errors)

# Conclusion gridsearch scénario 1
# ================================

# ---------------
# LogisticRegression
# 0.6770929814125659
# {'model__C': 0.1, 'model__solver': 'lbfgs'}
# 8.344 s
# Temps par observation : 0.0598 ms
# Erreurs critiques (2 -> 0) : 9
# Taux d'erreurs critiques (2 -> 0) : 0.1
# ---------------
# XGBoost
# 0.7006874715445617
# {'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__n_estimators': 300}
# 9.380 s
# Temps par observation : 0.1254 ms
# Erreurs critiques (2 -> 0) : 14
# Taux d'erreurs critiques (2 -> 0) : 0.15555555555555556
# ---------------
# lightGBM
# 0.7081279515053078
# {'model__learning_rate': 0.1, 'model__max_depth': 5, 'model__n_estimators': 100}
# 48.742 s
# Temps par observation : 0.1345 ms
# Erreurs critiques (2 -> 0) : 10
# Taux d'erreurs critiques (2 -> 0) : 0.1111111111111111

# Gains modestes : entre 0.008 et 0.011 de F1
# L'optimisation des paramètres seule ne permettra pas d'obtenir un haut score F1
# LightGBM est retenu comme modèle de référence du scénario 1

# | Modèle | F1 Macro (CV) | Erreurs critiques (2→0) | Taux d'erreurs critiques | Temps d'entraînement | Temps d'inférence | Explicabilité |
# |----------|----------:|----------:|----------:|----------:|----------:|----------|
# | Logistic Regression | 0.677 | **9** | **10.0 %** | **8.3 s** | **0.060 ms** | **Très élevée** |
# | XGBoost | 0.701 | 14 | 15.6 % | 9.4 s | 0.125 ms | Faible |
# | LightGBM | **0.708** | 10 | 11.1 % | 48.7 s | 0.135 ms | Faible à moyenne |

# XGBoost exclu à cause de son recall élevé sur la classe critique
# Logistic regression est le modèle le plus explicable et son temps d'entrainement est plus de 5x plus faible que celui de lightGBM
# lightGBM a le meilleur score F1 global et est très proche de la régression logistique pour le taux d'erreur critique. Explicabilité plus difficile, nécessité d'utiliser SHAP 

# ## Comparaison des modèles

# | Critère | Logistic Regression | LightGBM | Meilleur |
# |----------|----------:|----------:|----------|
# | F1 macro | 0.676 | **0.711** | LightGBM |
# | Temps d'entraînement | **9.8 s** | 58.0 s | Logistic Regression |
# | Temps d'inférence | **0.063 ms** | 0.111 ms | Logistic Regression |
# | Erreurs critiques (2→0) | **9** | 11 | Logistic Regression |
# | Taux d'erreurs critiques | **10.0 %** | 12.2 % | Logistic Regression |
# | Explicabilité intrinsèque | **Très élevée** | Moyenne | Logistic Regression |
# | Explicabilité avec SHAP | Faible intérêt (coefficients directement interprétables) | **Bonne** | LightGBM |
# | Répartition des facteurs de décision | Plus concentrée | **Plus distribuée** | LightGBM |
# | Risque de dépendance à une variable sensible | Plus élevé | **Semble plus faible** | LightGBM |

# ---

# ## Analyse comparative

# La régression logistique présente plusieurs avantages opérationnels. Elle est environ six fois plus rapide à entraîner, 
# offre le temps d'inférence le plus faible, produit moins d'erreurs critiques (9 contre 11) et possède une excellente 
# explicabilité intrinsèque grâce à l'interprétation directe des coefficients. 
# L'analyse des coefficients montre notamment que la variable sensible `nationalite_hors_ue` fait partie des variables 
# les plus influentes pour la prédiction de la classe de risque élevée.

# LightGBM obtient cependant les meilleures performances prédictives globales avec un F1 macro de **0.711** contre **0.676** 
# pour la régression logistique. L'analyse SHAP met également en évidence que ses décisions reposent sur un ensemble plus diversifié 
# de facteurs combinant données tabulaires et données textuelles (âge, niveau de diplôme, ancienneté, projet professionnel, 
# barrières identifiées, compétences, etc.). La variable sensible `nationalite_hors_ue` apparaît moins influente 
# dans les explications SHAP que les principales variables socio-professionnelles.

# L'analyse éthique ne permet pas de conclure à l'existence d'un biais discriminatoire direct. Néanmoins, 
# les résultats suggèrent que la régression logistique s'appuie davantage sur la variable sensible alors que LightGBM 
# répartit davantage la décision sur plusieurs facteurs explicatifs. Cette répartition plus équilibrée réduit potentiellement 
# le risque qu'une seule variable sensible domine la prédiction.

# ---

# ## Choix du modèle final

# Malgré un nombre légèrement plus élevé d'erreurs critiques et un coût computationnel supérieur, 
# **LightGBM est retenu comme modèle final**.

# Ce choix est motivé par :

# - le meilleur F1 macro global (**0.711**) ;
# - une prise en compte plus équilibrée des différentes sources d'information ;
# - l'exploitation simultanée des variables tabulaires et textuelles ;
# - une dépendance apparemment plus faible à la variable sensible `nationalite_hors_ue` ;
# - une explicabilité satisfaisante grâce à l'utilisation de SHAP.

# La régression logistique demeure toutefois une excellente baseline de référence, 
# notamment pour sa simplicité, sa rapidité d'exécution et sa forte transparence. 
# Dans un contexte où l'explicabilité ou la minimisation des erreurs critiques constituerait le critère principal, 
# elle pourrait également être retenue comme solution pertinente.

# En conclusion, **LightGBM offre le meilleur compromis entre performance prédictive, 
# explicabilité post-hoc et robustesse éthique**, ce qui justifie son utilisation pour la suite du projet.


# Conclusion scénario 2
# =====================
# L'analyse du scénario 2 montre que la suppression de la variable sensible entraîne une baisse mesurable des performances 
# prédictives pour les deux modèles. Cette baisse est plus importante pour LightGBM que pour la régression logistique, 
# ce qui indique que cette variable apportait une information utile à l'apprentissage.

# Cependant, l'analyse SHAP réalisée sur le scénario 1 montre que les décisions du modèle LightGBM reposaient principalement 
# sur l'âge, le niveau de diplôme, l'ancienneté professionnelle et les informations textuelles issues des synthèses d'entretien. 
# La variable sensible ne figurait pas parmi les facteurs dominants des prédictions individuelles.

# Par ailleurs, le nombre d'erreurs critiques reste inchangé pour LightGBM après suppression de la variable sensible (11 erreurs), 
# ce qui suggère que cette variable n'était pas la principale source des erreurs métier les plus graves.

# Ces résultats indiquent que la variable sensible contribuait à améliorer les performances globales du modèle 
# mais n'était pas le principal moteur des décisions ni des erreurs critiques observées.

# Les résultats indiquent aussi que SHAP ne rend pas compte des relations entre variable apprises par le modèle


# Conclusion scenario 3
# =====================

# le facteur limitant n'est plus l'algorithme mais l'information contenue dans le texte.

# L'analyse des erreurs du scénario NLP pur montre que les erreurs critiques concernent principalement des synthèses 
# contenant un vocabulaire très positif (« candidat dynamique », « profil autonome », « compétences techniques », 
# « aucun frein identifié »).

# Ces prédictions apparaissent cohérentes au regard du contenu textuel disponible. Les erreurs observées ne semblent 
# donc pas provenir d'une mauvaise compréhension du texte par le modèle, mais plutôt de l'absence d'informations 
# complémentaires présentes dans les données structurées du dossier.

# Cette observation suggère que les synthèses d'entretien, bien qu'informatives, ne permettent pas à elles seules d'expliquer 
# intégralement le risque de chômage longue durée. Les variables tabulaires apportent une information complémentaire 
# nécessaire à la prise de décision.

# Le scénario NLP pur démontre ainsi la richesse informationnelle des verbatims, tout en confirmant l'intérêt 
# d'une approche multimodale combinant texte et données structurées.