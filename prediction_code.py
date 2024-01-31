import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model\model (1).tflite1')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the MTCNN detector with adjusted parameters
detector = MTCNN(min_face_size=25, scale_factor=0.9)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define color for rectangle and text
rectangle_color = (0, 255, 0) 
text_color = (0, 255, 0) 

def predict_emotion(face):
    # Preprocess the face image (resize and normalize)
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0  # Normalize to [0, 1]
    face = np.expand_dims(face, axis=0)  
    face = np.expand_dims(face, axis=-1) 

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], face.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    emotion_prediction = interpreter.get_tensor(output_details[0]['index'])
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]

    # Return the emotion label (e.g., "Happy", "Sad", etc.)
    return emotion_label


# Load an image containing multiple faces
image_path = 'portrait-group-people_250469-11690.jpg'
image = cv2.imread(image_path)
# Check if image is loaded successfully
if image is None:
    print(f"Error: Unable to load the image from {image_path}")
    exit()

# Detect faces using MTCNN
faces = detector.detect_faces(image)

# Check if faces are detected
if not faces:
    print("No faces detected in the image.")
    exit()

# Initialize counters for each emotion
emotion_counts = {label: 0 for label in emotion_labels}

# Iterate through detected faces and predict emotions
for result in faces:
    x, y, width, height = result['box']
    face = image[y:y+height, x:x+width]

    # Check if the face region is valid
    if face.size == 0:
        print("Error: Invalid face region.")
        continue

    emotion_label = predict_emotion(face)

    # Draw a rectangle around the face with the emotion label
    cv2.rectangle(image, (x, y), (x+width, y+height), rectangle_color, 2)
    cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Update emotion counts
    emotion_counts[emotion_label] += 1

# Display the total counts or percentage for each emotion
total_faces = len(faces)
for label, count in emotion_counts.items():
    percentage = (count / total_faces) * 100 if total_faces > 0 else 0
    print(f"{label}: {count} ({percentage:.2f}%)")

# Save the image with emotion predictions
cv2.imwrite(image_path, image)

# Display the saved image with emotion predictions in Colab
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
