from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBClassifier

from data_preparation import build_preprocessor_scenario1, load_dataset_scenario1

models = {"random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
          "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42)}

X, y = load_dataset_scenario1("dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv")

X_trainval, X_unseen, y_trainval, y_unseen = train_test_split(X,
                                                              y,
                                                              test_size=0.20,
                                                              stratify=y,
                                                              random_state=42)

cv = StratifiedKFold(n_splits=5,
                     shuffle=True,
                     random_state=42)

# Optimisation scénario 1
# =======================
lr_params = {"model__C": [0.01, 0.1, 1, 10],
             "model__solver": ["lbfgs"]}

lr_pipeline = Pipeline([("prep", build_preprocessor_scenario1()),
                        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])

lr_search = GridSearchCV(estimator=lr_pipeline,
                         param_grid=lr_params,  
                         scoring="f1_macro",
                         cv=cv,
                         n_jobs=-1)

lr_search.fit(X_trainval, y_trainval)
 
xgb_params = {"model__n_estimators": [100, 300],
              "model__max_depth": [3, 5, 7],
              "model__learning_rate": [0.01, 0.05, 0.1]}

xgb_pipeline = Pipeline([("prep", build_preprocessor_scenario1()),
                         ("model", XGBClassifier(objective="multi:softprob", num_class=3, random_state=42, eval_metric="mlogloss"))])

xgb_search = GridSearchCV(estimator=xgb_pipeline,
                          param_grid=xgb_params,  
                          scoring="f1_macro",
                          cv=cv,
                          n_jobs=-1)

xgb_search.fit(X_trainval, y_trainval)

print("\n---------------\n")
print("LogisticRegression")
print(lr_search.best_score_)
print(lr_search.best_params_)

print("\n---------------\n")
print("XGBoost")
print(xgb_search.best_score_)
print(xgb_search.best_params_)

# Conclusion scénario tabulaire

# Une phase d'optimisation des hyperparamètres a été réalisée par validation croisée stratifiée à 5 folds 
# sur les deux modèles les plus performants issus du benchmark initial. 
# Le meilleur résultat est obtenu avec HistGradientBoostingClassifier (F1 macro = 0.629) qui semble avoir une marge 
# de progression supérieure au RandomForestClassifier (F1 macro = 0.605). 
# Le modèle HistGradientBoosting est retenu pour l'évaluation finale.