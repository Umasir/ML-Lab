import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load the Olivetti Faces dataset
data = fetch_olivetti_faces()
print("Data Shape:", data.data.shape)
print("Target Shape:", data.target.shape)
print("There are {} unique persons in the dataset".format(len(np.unique(data.target))))
print("Size of each image is {}x{}".format(data.images.shape[1], data.images.shape[2]))

# Function to display faces
def print_faces(images, target, top_n):
    top_n = min(top_n, len(images))
    grid_size = int(np.ceil(np.sqrt(top_n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)
    for i, ax in enumerate(axes.ravel()):
        if i < top_n:
            ax.imshow(images[i], cmap='bone')
            ax.axis('off')
            ax.text(2, 12, str(target[i]), fontsize=9, color='red')
            ax.text(2, 55, f"face: {i}", fontsize=9, color='blue')
        else:
            ax.axis('off')
    plt.show()

print_faces(data.images, data.target, 400)

# Function to display 40 unique individuals
def display_unique_faces(pics):
    fig = plt.figure(figsize=(24, 10))
    columns, rows = 10, 4
    for i in range(1, columns * rows + 1):
        img_index = 10 * i - 1
        if img_index < pics.shape[0]:
            img = pics[img_index, :, :]
            ax = fig.add_subplot(rows, columns, i)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Person {i}", fontsize=14)
            ax.axis('off')
    plt.suptitle("There are 40 distinct persons in the dataset", fontsize=24)
    plt.show()

display_unique_faces(data.images)

# Split the dataset
X = data.data
Y = data.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_gnb = gnb.predict(x_test)
gnb_accuracy = round(accuracy_score(y_test, y_pred_gnb) * 100, 2)
print("\n[GaussianNB Results]")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gnb))
print(f"Naive Bayes Accuracy: {gnb_accuracy}%")

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_mnb = mnb.predict(x_test)
mnb_accuracy = round(accuracy_score(y_test, y_pred_mnb) * 100, 2)
print("\n[MultinomialNB Results]")
print(f"Multinomial Naive Bayes Accuracy: {mnb_accuracy}%")

# Misclassified images (MultinomialNB)
misclassified_idx = np.where(y_pred_mnb != y_test)[0]
num_misclassified = len(misclassified_idx)
print(f"Number of misclassified images: {num_misclassified}")
print(f"Total images in test set: {len(y_test)}")
print(f"Accuracy: {round((1 - num_misclassified / len(y_test)) * 100, 2)}%")

# Show some misclassified images
n_misclassified_to_show = min(num_misclassified, 5)
plt.figure(figsize=(10, 5))
for i in range(n_misclassified_to_show):
    idx = misclassified_idx[i]
    plt.subplot(1, n_misclassified_to_show, i + 1)
    plt.imshow(x_test[idx].reshape(64, 64), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_mnb[idx]}")
    plt.axis('off')
plt.show()

# ROC AUC score for MultinomialNB
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_prob = mnb.predict_proba(x_test)

print("\n[AUC Scores per class]")
for i in range(y_test_bin.shape[1]):
    try:
        roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])
        print(f"Class {i} AUC: {roc_auc:.2f}")
    except ValueError:
        print(f"Class {i} AUC: Undefined (only one class present in y_true)")
