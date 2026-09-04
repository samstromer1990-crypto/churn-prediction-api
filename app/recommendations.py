


def generate_recommendation(customer_data: dict, risk_category: str) -> str:
    """
    customer_data: the same dict passed to ChurnPredictor.predict()
    risk_category: "Low" / "Medium" / "High", from the prediction result
    """
    notes = []

    if risk_category == "High":
        notes.append(
            "High risk: recommend immediate, personalized retention outreach "
            "(e.g. a retention offer or discount)."
        )
    elif risk_category == "Medium":
        notes.append(
            "Medium risk: recommend a proactive check-in and highlighting "
            "underused value-add services."
        )
    else:
        notes.append("Low risk: no immediate action needed; continue standard engagement.")

    if risk_category != "Low":
        if customer_data.get("Contract") == "Month-to-month":
            notes.append(
                "Customer is on a month-to-month contract, the strongest churn driver "
                "identified in this project -- consider an incentive to move them to "
                "a longer-term contract."
            )

        if customer_data.get("InternetService") == "Fiber optic":
            notes.append(
                "Customer has Fiber optic internet, which shows elevated churn in "
                "this dataset -- consider a service satisfaction check-in."
            )

    return " ".join(notes)