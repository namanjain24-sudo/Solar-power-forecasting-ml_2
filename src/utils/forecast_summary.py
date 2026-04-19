def generate_summary(predictions):
    if not predictions or len(predictions) == 0:
        return "No prediction data available."

    summary = ""

    avg = sum(predictions) / len(predictions)

    # thresholds (you can tune later)
    threshold_high = 70
    threshold_low = 30

    if avg > threshold_high:
        summary += "High solar generation expected.\n"
    elif avg < threshold_low:
        summary += "Low solar generation expected.\n"
    else:
        summary += "Moderate solar generation expected.\n"

    # variability
    variability = max(predictions) - min(predictions)

    if variability > 40:
        summary += "High variability detected.\n"
    else:
        summary += "Stable generation pattern.\n"

    # optional extra insight
    peak = max(predictions)
    low = min(predictions)

    summary += f"Peak generation: {peak}\n"
    summary += f"Lowest generation: {low}\n"

    return summary