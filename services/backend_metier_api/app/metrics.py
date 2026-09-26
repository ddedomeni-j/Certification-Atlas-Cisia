from prometheus_client import Counter

FEEDBACKS_TOTAL = Counter(
    "feedbacks_total",
    "Nombre de feedbacks enregistrés par classe corrigée.",
    labelnames=("corrected_class",),
)


def observe_feedback(corrected_class: int) -> None:
    FEEDBACKS_TOTAL.labels(corrected_class=str(corrected_class)).inc()