flowchart TD
    C[Conseiller]
    O[Équipe MLOps]
    C ~~~ O
    F[Frontend Streamlit]

    subgraph BACK["Backend métier"]
        direction TB
        B[API métier]
        DB[(Base historique<br/>inférences + feedbacks)]
        B -->|Journalise prédiction| DB
        B -->|Enregistre feedback| DB
    end

    subgraph MODEL["Service de prédiction"]
        direction TB
        M[API modèle<br/>S5 LightGBM]
        A[(Artefact modèle actif)]
        A -->|Charge| M
    end

    subgraph RETRAIN["Service de réentraînement"]
        direction TB
        T[API de réentraînement]
    end

    subgraph MON["Monitoring"]
        direction TB
        PR[Prometheus]
        G[Grafana]
        PR --> G
    end

    subgraph OPT["Optimisation continue"]
        direction TB
        R[Réentraînement<br/>modèle candidat]
        E[Évaluation<br/>jeu de référence + feedbacks]
        ML[MLflow<br/>runs, métriques, versions]
        P[Validation humaine<br/>et promotion]

        R --> E --> ML --> P
    end

    C -->|Saisie dossier<br/>feedback / correction| F
    F -->|Appels métier| B
    B -->|Résultat de prédiction| F
    F -->|Restitution| C

    B -->|POST /predict| M
    M -->|Classe, probabilités,<br/>décision, version| B

    B -->|Logs, latence, erreurs| PR
    M -->|Métriques de prédiction| PR

    O -->|Lance le réentraînement| F
    F -->|POST /retraining| T
    T -->|Lit les feedbacks non utilisés| DB
    T --> R

    P -->|Version validée| A

    classDef user fill:#FDE68A,stroke:#B45309,color:#451A03;
    classDef frontend fill:#D1FAE5,stroke:#047857,color:#064E3B;
    classDef service fill:#DBEAFE,stroke:#2563EB,color:#172554;
    classDef storage fill:#F3E8FF,stroke:#7E22CE,color:#3B0764;
    classDef monitoring fill:#FCE7F3,stroke:#BE185D,color:#500724;
    classDef mlops fill:#FFEDD5,stroke:#C2410C,color:#431407;

    class C,O user;
    class F frontend;
    class B,M,T service;
    class DB,A storage;
    class PR,G monitoring;
    class R,E,ML,P mlops;

    style BACK fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style MODEL fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style RETRAIN fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style MON fill:#FDF2F8,stroke:#BE185D,stroke-width:2px
    style OPT fill:#FFF7ED,stroke:#C2410C,stroke-width:2px