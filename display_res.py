import numpy as np
import matplotlib.pyplot as plt

# Load the saved data
data = np.load('test_example_epochX.npz')  # Replace 'X' with the epoch number
img = data['img']
label = data['label']
prediction = data['prediction']
loss = data['loss']

# Example: Plot the first image, label, and prediction
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(img[0], cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(label[0], cmap='gray')
plt.title('True Label')
plt.subplot(133)
plt.imshow(prediction[0], cmap='gray')
plt.title('Predicted Image')
plt.show()