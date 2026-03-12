import torch
import torch.nn.functional as F
from model_loader import load_model, load_feature_extractor, load_fusion_model
from utils import clinical_to_tensor, genomic_to_tensor, generate_report, CLASSES

# Pre-load models to memory for efficiency
model = load_model()
feature_extractor = load_feature_extractor()
fusion_model = load_fusion_model()

LABEL_MAP = CLASSES
TUM_IDX = LABEL_MAP.index("TUM")
STR_IDX = LABEL_MAP.index("STR")

def run_prediction(image_tensor, age, gender, stage, gene_score, genomic_risk):
    """
    Runs inference using both the base CNN and the multi-modal Fusion model.
    Returns the most confident prediction and diagnostic metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models and tensors to device
    image_tensor = image_tensor.to(device)
    model.to(device)
    feature_extractor.to(device)
    fusion_model.to(device)

    model.eval()
    feature_extractor.eval()
    fusion_model.eval()

    # Convert clinical and genomic data for the fusion model
    clinical = clinical_to_tensor(age, gender, stage).to(device)
    genomic = genomic_to_tensor([gene_score, genomic_risk]).to(device)

    with torch.no_grad():
        # 1. Base ResNet50 Prediction (Visual Only)
        base_logits = model(image_tensor)
        base_probs = F.softmax(base_logits, dim=1)
        base_pred = int(torch.argmax(base_probs, dim=1).item())
        base_conf = float(base_probs[0, base_pred].item())

        # 2. Fusion Model Prediction (Visual + Clinical + Genomic)
        img_features = feature_extractor(image_tensor)
        img_features = img_features.view(img_features.size(0), -1)
        fusion_logits = fusion_model(img_features, clinical, genomic)
        fusion_probs = F.softmax(fusion_logits, dim=1)
        fusion_pred = int(torch.argmax(fusion_probs, dim=1).item())
        fusion_conf = float(fusion_probs[0, fusion_pred].item())

        # Logic to decide which model to trust based on confidence
        FUSION_THRESHOLD = 0.30
        if fusion_conf >= FUSION_THRESHOLD:
            pred_idx = fusion_pred
            pred_confidence = fusion_conf
            use_probs = fusion_probs
            used_fusion = True
        else:
            # Fallback to base model if fusion weights are untrained/uncertain
            pred_idx = base_pred
            pred_confidence = base_conf
            use_probs = base_probs
            used_fusion = False

    label = LABEL_MAP[pred_idx]

    # Calculate risk score based on Tumor and Stroma probabilities
    tum_prob = float(use_probs[0, TUM_IDX].item())
    str_prob = float(use_probs[0, STR_IDX].item())
    cancer_risk_score = tum_prob + str_prob

    if cancer_risk_score >= 0.45:
        risk = "high"
    elif cancer_risk_score >= 0.20:
        risk = "medium"
    else:
        risk = "low"

    # Map all class probabilities for the frontend charts
    probabilities = {LABEL_MAP[i]: float(use_probs[0, i].item()) for i in range(len(LABEL_MAP))}

    # Generate the text-based clinical report
    report = generate_report(pred_idx, pred_confidence)

    return {
        "label": label,
        "prob": pred_confidence,
        "risk": risk,
        "risk_score": round(cancer_risk_score * 100),
        "probabilities": probabilities,
        "predicted_class": label,
        "diagnosis": report["diagnosis"],
        "tier": report["tier"],
        "recommendation": report["recommendation"],
        "details": report["details"],
        "confidence": pred_confidence,
        "used_fusion": used_fusion,
        "clinical_tensor": clinical, # Useful for SHAP explainer
        "genomic_tensor": genomic    # Useful for SHAP explainer
    }