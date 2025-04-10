import os
import pickle
import cv2
import mediapipe as mp

# Suppress MediaPipe/TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.5  # Increased confidence threshold
)

DATA_DIR = './data'

data = []
labels = []
skipped_images = 0

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    if not os.path.isdir(dir_path):
        continue
        
    for img_path in os.listdir(dir_path):
        try:
            # Skip non-image files
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path_full = os.path.join(dir_path, img_path)
            img = cv2.imread(img_path_full)
            
            if img is None:
                print(f"Could not read image: {img_path_full}")
                skipped_images += 1
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if not results.multi_hand_landmarks:
                print(f"No hands detected in: {img_path_full}")
                skipped_images += 1
                continue
                
            data_aux = []
            x_ = []
            y_ = []

            # Visualize detections
            annotated_img = img.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Collect landmarks
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

            # Normalize coordinates
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            
            # Show detection (press any key to continue)
            cv2.imshow('Hand Detection', annotated_img)
            if cv2.waitKey(200) == ord('q'):  # Press 'q' to quit early
                break
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            skipped_images += 1
            continue

# Save data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Clean up
cv2.destroyAllWindows()
hands.close()

# Print summary
print(f"\nDataset creation complete:")
print(f"- Successfully processed: {len(data)} images")
print(f"- Skipped: {skipped_images} images")
print(f"- Unique labels: {set(labels)}")
print(f"Data saved to data.pickle")
