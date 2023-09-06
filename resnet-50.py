import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Load the image and preprocess it
img_path = 'Detecting-Images-1/test/stage 1/Stage-1-Calcium-deficiency-update-46-_JPG.rf.e58f7e5a29b3680c8a94e23770840331.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make a prediction with the model
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=1)[0][0]

# Load the image again and draw a bounding box around the predicted object
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
cv2.rectangle(img, (0, 0), (224, 224), (0, 255, 0), 2)
cv2.putText(img, f'{decoded_preds[1]}: {decoded_preds[2]:.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
