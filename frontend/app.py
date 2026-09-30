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
    review_status = result["review_status"]
    
    st.subheader("Résultat de la prédiction")
    st.markdown(f":{color}[**{result['decision']}**]")

    st.info("Cette estimation est une aide à la décision. Vérifiez les freins "
            "périphériques, le parcours professionnel et les éléments de l'entretien "
            "avant toute orientation.")
    
    if review_status == "abstention":
        st.error("Analyse complémentaire requise : les probabilités sont insuffisantes "
                 "pour formuler une recommandation fiable.")
    elif review_status == "reinforced_review":
        st.warning("Decision à confirmer par un conseiller : "
                   "la confiance du modèle est intermédiaire.")

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
    st.title("Certification Atlas CISIA")
    st.caption("Aide à la décision pour l'accompagnement vers le retour à l'emploi")
    
    st.subheader("Modèle utilisé")
    st.write("Le système utilise le scénario S5 avec le modèle LightGBM. "
             "Il exploite les données tabulaires et la synthèse d'entretien comme une variable catégorielle.")
    
    st.info("La prédiction est une aide à l'analyse. "
            "Le conseiller conserve la décision finale et peut l'enrichir par un feedback.")

    prediction, feedback, monitoring, administration = st.columns(4)

    with prediction:
        st.subheader("1. Évaluer")
        st.markdown("### 📋")
        st.write("Saisissez le dossier et la synthèse d'entretien pour obtenir "
                 "une estimation du délai de retour à l'emploi.")
        st.markdown("- Classe et probabilités\n"
                    "- Recommandation métier\n"
                    "- Version du modèle")

    with feedback:
        st.subheader("2. Rechercher & corriger")
        st.markdown("### 🔎")
        st.write("Retrouvez une prédiction journalisée ; enregistrez "
                 "une correction lorsque la situation réelle est connue.")
        st.markdown("- Historique\n"
                    "- Classe corrigée\n"
                    "- Commentaire conseiller")

    with monitoring:
        st.subheader("3. Superviser")
        st.markdown("### 📈")
        st.write("Consultez l'état des services et le suivi des expérimentations.")
        st.markdown("- Grafana : disponibilité, erreurs et latence\n"
                    "- MLflow : métriques des modèles réentrainés")

    with administration:
        st.subheader("4. Améliorer")
        st.markdown("### 🔄⚙️")
        st.write("Espace réservé à l'équipe MLOps pour préparer un modèle candidat en utilisant "
                 "les feedbacks des conseillers. La promotion des modèles candidats nécessite une validation humaine.")

        st.markdown("- Réentraînement contrôlé\n"
                    "- Évaluation sur référence\n"
                    "- Validation humaine avant promotion")


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
            f"revue : {item['review_status']} | "
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
    st.title("Boucle d'amélioration MLOps")
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
    [
        "Accueil",
        "1. Évaluer un dossier",
        "2. Rechercher et corriger",
        "3. Superviser",
        "4. Boucle d'amélioration MLOps",
    ],
)

if page == "Accueil":
    home_page()
elif page == "1. Évaluer un dossier":
    prediction_page()
elif page == "2. Rechercher et corriger":
    feedback_page()
elif page == "3. Superviser":
    monitoring_page()
else:
    administration_ml_page()