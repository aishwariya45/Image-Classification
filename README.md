## Convolutional Deep Neural Network for Image Classification
### AIM
To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and DataseT
The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the FashionMNIST dataset. The FashionMNIST dataset contains grayscale images of 10 different clothing categories (e.g., T-shirt, trousers, dress, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while ensuring computational efficiency.

## Neural Network Model

<img width="1063" height="625" alt="image" src="https://github.com/user-attachments/assets/00375475-52d2-4da5-a72b-8281739c3d61" />
## DESIGN STEPS
STEP 1: Problem Statement
Define the goal of classifying fashion apparel items into 10 categories using a CNN. Ensure high accuracy while maintaining efficiency.

STEP 2: Dataset Collection
Use the FashionMNIST dataset, which contains 60,000 training and 10,000 test images of clothing items. Each image is 28×28 grayscale and labeled with one of 10 classes.

STEP 3: Data Preprocessing
Convert images to tensors and normalize pixel values to [0,1]. Use DataLoaders for efficient batch processing during training and testing.

STEP 4: Model Architecture
Design a CNN with convolutional layers for feature extraction, ReLU activation, pooling layers for downsampling, and fully connected layers for classification.

STEP 5: Model Training
Train the CNN using CrossEntropyLoss and Adam optimizer for multiple epochs. Monitor accuracy and loss to ensure proper learning.

STEP 6: Model Evaluation
Test the model on unseen FashionMNIST data, compute performance metrics, and analyze using a confusion matrix.

STEP 7: Model Deployment & Visualization
Save the trained model for future use and visualize predictions on sample test images. Optionally, integrate into an application for real-world use.
## PROGRAM
## NAME:S.AISHWARIYA
## REG NO : 212224240005

```
class CNNClassifier(nn.Module):
  def __init__(self): # Define __init__ method explicitly
    super(CNNClassifier, self).__init__() # Call super().__init__() within __init__
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # Correct argument names
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Correct argument names
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Correct argument names
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjust input size for Linear layer (Calculation needs update if image size changed)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x))) # Correctly call self.conv1
    x = self.pool(torch.relu(self.conv2(x)))  # Correctly call self.conv2
    x = self.pool(torch.relu(self.conv3(x))) # Correctly call self.conv3
    x = x.view(x.size(0), -1) # Flatten the tensor
    x = torch.relu(self.fc1(x)) # Correctly call self.fc1
    x = torch.relu(self.fc2(x)) # Correctly call self.fc2
    x = self.fc3(x)
    return x


```



```


# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)



```





```

# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    print('Name: AISHWARIYA S')
    print('Register Number: 212224240005')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print only once per epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


```
## OUTPUT
  Training Loss per Epoch
<img width="272" height="122" alt="image" src="https://github.com/user-attachments/assets/f05c20ca-737a-4f36-9974-7ff8fdb6eae5" />
 Confusion Matrix
<img width="742" height="688" alt="image" src="https://github.com/user-attachments/assets/a611a7e4-b30a-4f75-bd95-397e11a71788" />

 Classification Report
<img width="564" height="364" alt="image" src="https://github.com/user-attachments/assets/5292edd4-9807-42e4-9a7f-02a995fba4c9" />
  New Sample Data Prediction
 <img width="483" height="497" alt="image" src="https://github.com/user-attachments/assets/6875b7d7-7ab6-4c22-9588-a75f12d5dfb0" />
