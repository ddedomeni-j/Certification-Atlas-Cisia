from __future__ import annotations

import argparse
import json
import shutil

from config import ACTIVE_MODEL_PATH, CANDIDATE_VERSION, MODELS_DIR, REPORTS_DIR


def promote_candidate(version: str) -> None:
    decision_path = REPORTS_DIR / "promotion_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    if not decision["promote"]:
        raise RuntimeError(
            "Promotion refusée par la recommandation. "
            "Aucun artefact n'a été modifié."
        )

    candidate_model_path = MODELS_DIR / f"pipeline_{CANDIDATE_VERSION}.joblib"
    candidate_metadata_path = MODELS_DIR / f"pipeline_{CANDIDATE_VERSION}.json"
    promoted_model_path = MODELS_DIR / f"pipeline_{version}.joblib"
    promoted_metadata_path = MODELS_DIR / f"pipeline_{version}.json"

    if promoted_model_path.exists() or promoted_metadata_path.exists():
        raise FileExistsError(f"La version {version} existe déjà.")

    if not candidate_model_path.exists():
        raise FileNotFoundError(f"Artefact candidat absent : {candidate_model_path}")

    if not candidate_metadata_path.exists():
        raise FileNotFoundError(
            f"Métadonnées candidat absentes : {candidate_metadata_path}"
        )

    shutil.copy2(candidate_model_path, promoted_model_path)

    metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
    metadata["model_version"] = version
    metadata["artifact"] = promoted_model_path.name

    promoted_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ACTIVE_MODEL_PATH.write_text(
        json.dumps({"active_version": version}, indent=2),
        encoding="utf-8",
    )

    print(f"Candidat promu vers {version}.")
    print(f"Modèle actif : {version}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        required=True,
        help="Version de production à créer, par exemple v4.",
    )
    arguments = parser.parse_args()

    promote_candidate(arguments.version)


if __name__ == "__main__":
    main()