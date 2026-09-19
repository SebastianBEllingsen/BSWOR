from ultralytics.models.sam import SAM3SemanticPredictor
import cv2

# Initialize predictor with configuration
overrides = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "model": "sam3.pt",
    "quantize": 16,  # Use FP16 for faster inference
    "save": True,
}
predictor = SAM3SemanticPredictor(overrides=overrides)

cam = cv2.VideoCapture(1)

ret, frame = cam.read()

#count = 0
#cv2.imwrite("frame%d.jpg" % count, frame)
#count +=1

# Set image once for multiple queries
predictor.set_image(frame)



predictor.set_

# Query with multiple text prompts
results = predictor(text=["person", "glasses"])

# Works with descriptive phrases
results = predictor(text=["person with red cloth", "person with blue cloth"])

# Query with a single concept
results = predictor(text=["a person"])