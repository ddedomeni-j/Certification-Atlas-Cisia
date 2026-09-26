from prometheus_client import Counter

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Nombre de prédictions servies par classe prédite.",
    labelnames=("predicted_class",),
)


def observe_prediction(prediction: int) -> None:
    MODEL_PREDICTIONS_TOTAL.labels(predicted_class=str(prediction)).inc()