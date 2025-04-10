import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_classifier():
    # Load data
    try:
        with open('./data.pickle', 'rb') as f:
            data_dict = pickle.load(f)
    except FileNotFoundError:
        print("Error: data.pickle not found")
        return

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    
    # Check for classes with <2 samples
    from collections import Counter
    label_counts = Counter(labels)
    print("Current samples per class:", label_counts)
    
    # Remove classes with <2 samples
    valid_indices = [i for i, label in enumerate(labels) if label_counts[label] >= 2]
    data = data[valid_indices]
    labels = labels[valid_indices]
    
    # Now split (without stratify)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, 
        test_size=0.2, 
        shuffle=True
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    
    print('\nClassification Report:')
    print(classification_report(y_test, y_predict))
    print(f'\nAccuracy: {score*100:.2f}%')

    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)

if __name__ == '_main_':
    train_classifier()
