from loguru import logger
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report
)


REVENUE_PER_CUSTOMER = 8000      # ₪ average ARR per customer
RETENTION_COST = 500             # ₪ cost of outreach/offer to save a customer
FALSE_NEGATIVE_COST = REVENUE_PER_CUSTOMER   # missed churner = lost revenue
FALSE_POSITIVE_COST = RETENTION_COST         # unnecessary outreach


def business_cost(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * FALSE_NEGATIVE_COST) + (fp * FALSE_POSITIVE_COST)
    saved_revenue = tp * REVENUE_PER_CUSTOMER
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "cost_of_misses_ils": int(fn * FALSE_NEGATIVE_COST),
        "cost_of_false_alarms_ils": int(fp * FALSE_POSITIVE_COST),
        "total_cost_ils": int(total_cost),
        "saved_revenue_ils": int(saved_revenue),
    }


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    biz = business_cost(y_test, y_pred)

    results = {
        "model": model_name,
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        **biz
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"F1: {f1:.4f} | AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    logger.info(f"Saved revenue: ₪{biz['saved_revenue_ils']:,}")
    logger.info(f"Total cost of errors: ₪{biz['total_cost_ils']:,}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return results