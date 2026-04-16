def get_label(score):

    if score > 25:
        return "High Risk 🔴"
    elif score > 12:
        return "Medium Risk 🟡"
    else:
        return "Low Risk 🟢"