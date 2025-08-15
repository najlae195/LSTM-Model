import shap

def compute_shap_deep(model, X_background, X_sample):
    explainer = shap.DeepExplainer(model, X_background)
    shap_vals = explainer.shap_values(X_sample, check_additivity=False)[0]
    return shap_vals

