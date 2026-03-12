import torch
import numpy as np
from PIL import Image
import os
from uuid import uuid4
import threading
from utils import preprocess_image

try:
    import cv2
except ImportError:
    cv2 = None

gradcam_lock = threading.Lock()

def generate_gradcam(model, image_path, clinical=None, genomic=None):
    """
    Generates a Grad-CAM activation map. 
    If model is a FusionModel, it hooks into the underlying ResNet50 backbone.
    """
    if cv2 is None:
        return None

    with gradcam_lock:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        gradients = []
        activations = []

        # Determine if we are looking at a FusionModel or base ResNet50
        # If FusionModel, we need to find where the ResNet50 backbone is stored
        is_fusion = hasattr(model, "clinical_net")
        
        # In your model_loader.py, the FusionModel doesn't explicitly store the 
        # backbone as a named attribute, but uses a separate feature_extractor.
        # For Grad-CAM to work on Fusion, we hook the last layer of the CNN part.
        if is_fusion:
            # This assumes you pass the feature_extractor or the base model 
            # specifically for the visual explanation part.
            from model_loader import load_model
            target_model = load_model().to(device)
        else:
            target_model = model

        target_layer = target_model.layer4

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        # Process image
        img_tensor = preprocess_image(image_path).to(device).requires_grad_(True)

        # Forward pass
        output = target_model(img_tensor)
        pred_class = int(torch.argmax(output, dim=1).item())
        loss = output[0, pred_class]

        # Backward pass
        target_model.zero_grad()
        loss.backward()

        # Extract gradients and activations
        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]

        # Global Average Pooling of gradients
        weights = np.mean(grads, axis=(1, 2))

        # Create Heatmap
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Overlay on original image
        img = np.array(Image.open(image_path).convert("RGB").resize((512, 512)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

        # Save result
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        save_dir = os.path.join(base_dir, "frontend", "static", "uploads")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"gradcam_{uuid4().hex}.png"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, overlay)

        # Cleanup
        forward_handle.remove()
        backward_handle.remove()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return f"/static/uploads/{filename}"