from __future__ import annotations

from datetime import datetime
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8001")
RETRAINING_API_URL=os.getenv("RETRAINING_API_URL", "http://127.0.0.1:8002")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://127.0.0.1:5000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000")

DECISION_COLORS = {
    0: "green",
    1: "orange",
    2: "red",
}

st.set_page_config(
    page_title="Atlas CISIA",
    page_icon="A",
    layout="wide",
)

def call_backend(
    method: str,
    path: str,
    **kwargs,
) -> requests.Response | None:
    """Appelle le backend métier et affiche les erreurs HTTP à l'utilisateur."""
    try:
        response = requests.request(
            method,
            f"{BACKEND_URL}{path}",
            timeout=10,
            **kwargs,
        )
    except requests.RequestException:
        st.error("Le backend métier est indisponible. Démarrez-le sur le port 8001.")
        return None

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Erreur API ({response.status_code}) : {detail}")
        return None

    return response


def show_prediction(result: dict) -> None:
    """Affiche une prédiction métier et ses probabilités."""
    prediction = result["prediction"]
    color = DECISION_COLORS[prediction]

    st.subheader("Résultat de la prédiction")
    st.markdown(
        f":{color}[**{result['decision']}**]"
    )

    first, second, third = st.columns(3)
    first.metric("Classe prédite", prediction)
    second.metric("Version modèle", result["model_version"])
    third.metric("Request ID", result["request_id"][:8])

    st.write("Probabilités par classe")
    probabilities = {
        f"Classe {class_label}": probability
        for class_label, probability in result["probabilities"].items()
    }
    st.bar_chart(probabilities, horizontal=True)

    with st.expander("Détails techniques"):
        st.json(result)


def home_page() -> None:
    st.title("Atlas CISIA")
    st.subheader("Aide à la décision pour le retour à l'emploi")

    st.write(
        "Cette application est l'API client d'un système d'IA "
        "d'aide à la décision utilisant un modèle de prédiction du délai de retour à l'emploi "
        "de personnes à partir des informations du dossier et de la synthèse d'entretien."
    )

    st.markdown(
        """
        **Chaîne de traitement**

        1. Le premier onglet permet au conseiller :
            - de saisir les éléments du dossier de la personne faisant l'objet de l'évaluation.
            - d'appeler le service de prédiction pour obtenir une aide à la décision qui peut être
                - une prédiction de délai de retour à l'emploi
                - une indication de personne protégée nécessitant une validation humaine
                - une impossibilité à prédire un résultat
            - Les seuils chosis pour la décision sont ...
        2. Le deuxieme onglet permet au conseiller de saisir un feedback
            - en retrouvant la personne évaluée grace à son identifiant
            - en éditant et en corrigeant la décision en fonction de sa réalisation
        3. Le troisième onglet permet d'accéder aux tableaux de suivi Grafana et MlFlow

        Remarques : 
        - Le scénario retenu pour le modèle est S5 LightGBM (toutes les variables utilisées et le texte utilisé comme variable catégorielle)
        - La prédiction est journalisée avec la version du modèle utilisée.
        - Les feedbacks alimentent la boucle de réentraînement et de promotion d'un nouveau modèle.
        """
    )


def prediction_page() -> None:
    st.title("Prédiction")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Âge", min_value=16, max_value=100, value=35, step=1)
            anciennete = st.number_input(
                "Ancienneté dans le poste (années)",
                min_value=0.0,
                max_value=60.0,
                value=2.5,
                step=0.1,
            )
            niveau_diplome = st.selectbox(
                "Niveau de diplôme",
                ["Sans diplôme", "Bac", "Bac+2", "Bac+5"],
            )
            est_allocataire = st.selectbox(
                "Est allocataire",
                options=[0.0, 1.0],
                format_func=lambda value: "Oui" if value == 1.0 else "Non",
            )

        with right:
            nationalite_hors_ue = st.selectbox(
                "Nationalité hors UE",
                options=[0, 1],
                format_func=lambda value: "Oui" if value == 1 else "Non",
            )
            code_insee_commune = st.text_input("Code INSEE de la commune",
                                               value="75056",
                                               max_chars=5)

            code_rome_vise = st.text_input("Code ROME visé",
                                           value="M1805",
                                           max_chars=5)

            SYNTHESIS_OPTIONS = ["Compétences à réactualiser sur les outils numériques. Motivation correcte.",
                                 "Cumul de difficultés. Pas de moyen de transport et garde d'enfants complexe.",
                                 "Excellente présentation, compétences techniques à jour. Reprise rapide visée.",
                                 "Projet de reconversion à affiner. Besoin d'une formation complémentaire.",
                                 "Problème ponctuel de mobilité géographique. Zone mal desservie.",
                                 "Candidat très dynamique, mobilité nationale validée. Projet clair.",
                                 "Profil autonome, recherche active, aucun frein périphérique identifié.",
                                 "Perte de confiance importante. Risque d'exclusion, barrière de la langue.",
                                 "Freins périphériques majeurs. Situation d'illettrisme numérique constatée."]

        synthese = st.selectbox("Synthèse d'entretien", options=SYNTHESIS_OPTIONS)
        
        submitted = st.form_submit_button("Lancer la prédiction")
        if submitted:
            payload = {"age": age,
                    "anciennete_poste_ans": anciennete,
                    "niveau_diplome": niveau_diplome,
                    "est_allocataire": est_allocataire,
                    "nationalite_hors_ue": nationalite_hors_ue,
                    "code_insee_commune": code_insee_commune.strip().upper(),
                    "code_rome_vise": code_rome_vise.strip().upper(),
                    "synthese_entretien": synthese}

            with st.spinner("Évaluation du dossier..."):
                response = call_backend("POST",
                                        "/applications/predict",
                                        json=payload)

            if response is not None:
                result = response.json()
                st.session_state["last_prediction"] = result
                show_prediction(result)


def feedback_page() -> None:
    st.title("Feedback conseiller")

    response = call_backend("GET", "/applications")
    if response is None:
        return

    applications = response.json()
    if not applications:
        st.info("Aucune prédiction journalisée. Lancez d'abord une prédiction.")
        return

    labels = {
        item["request_id"]: (
            f"{item['request_id'][:8]} | {item['decision']} | "
            f"{item['created_at'][:19]}"
        )
        for item in applications
    }

    request_id = st.selectbox(
        "Prédiction à corriger",
        options=list(labels),
        format_func=lambda identifier: labels[identifier],
    )

    detail_response = call_backend("GET", f"/applications/{request_id}")
    if detail_response is None:
        return

    prediction = detail_response.json()
    show_prediction(prediction)

    with st.form("feedback_form"):
        corrected_class = st.selectbox(
            "Classe corrigée",
            options=[0, 1, 2],
            format_func=lambda value: {
                0: "0 - Retour rapide à l'emploi",
                1: "1 - Retour à l'emploi intermédiaire",
                2: "2 - Risque de chômage de longue durée",
            }[value],
        )
        comment = st.text_area("Commentaire", max_chars=1000)
        submitted = st.form_submit_button("Enregistrer le feedback")

    if submitted:
        feedback_response = call_backend(
            "POST",
            "/feedback",
            json={
                "request_id": request_id,
                "corrected_class": corrected_class,
                "comment": comment or None,
            },
        )

        if feedback_response is not None:
            result = feedback_response.json()
            created_at = datetime.fromisoformat(result["created_at"])
            st.success(
                f"Feedback #{result['feedback_id']} enregistré "
                f"le {created_at.strftime('%d/%m/%Y à %H:%M')}."
            )

def administration_ml_page() -> None:
    st.title("Administration ML")
    st.caption("Espace réservé à l'équipe MLOps / model_admin.")

    st.warning(
        "Cette action prépare un nouveau candidat à partir des feedbacks non utilisés, "
        "l'évalue contre le modèle actif et produit une recommandation. "
        "Elle ne promeut jamais automatiquement un modèle."
    )

    if st.button("Lancer le réentraînement", type="primary"):
        try:
            with st.spinner("Réentraînement, évaluation et recommandation en cours..."):
                response = requests.post(
                    f"{RETRAINING_API_URL}/retraining",
                    timeout=300,
                )
        except requests.RequestException:
            st.error(
                "Le service de réentraînement est indisponible. "
                "Vérifiez le service Docker sur le port 8002."
            )
            return

        if response.status_code == 422:
            st.info(response.json().get("detail", "Aucun feedback disponible."))
            return

        if not response.ok:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")
            return

        result = response.json()
        st.success(
            f"{result['feedback_count']} feedback(s) ont été intégrés au candidat "
            f"{result['candidate_version']}."
        )
        st.write(f"Recommandation : {result['recommendation']}")
        st.link_button("Consulter MLflow", MLFLOW_URL)

def monitoring_page() -> None:
    st.title("Suivi technique")
    st.write(
        "Accédez aux outils de suivi de l'API du modèle et de la boucle d'amélioration."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("MLflow")
        st.write("Runs, paramètres, métriques et artefacts des candidats.")
        st.link_button(
            "Ouvrir MLflow",
            MLFLOW_URL,
            use_container_width=True,
        )

    with right:
        st.subheader("Grafana")
        st.write("Métriques, latence, erreurs et activité des services.")
        st.link_button(
            "Ouvrir Grafana",
            GRAFANA_URL,
            use_container_width=True,
        )

page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Prédiction", "Feedback", "Suivi technique", "Administration ML"],
)


if page == "Accueil":
    home_page()
elif page == "Prédiction":
    prediction_page()
elif page == "Feedback":
    feedback_page()
elif page == "Suivi technique":
    monitoring_page()
else:
    administration_ml_page()