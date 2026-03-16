# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a **Convolutional Neural Network (CNN)** for image classification using the **Fashion-MNIST** dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve **high accuracy** while maintaining **efficiency**.
## Neural Network Model

![425840736-6acab57a-cf5e-4963-a584-024b1d03e3e9](https://github.com/user-attachments/assets/2a4ae2ee-eccd-4f09-8dcb-19aa85efc78a)

## DESIGN STEPS

#### STEP 1: Problem Statement  
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a **Convolutional Neural Network (CNN)**.  

#### STEP 2: Dataset Collection  
Use the **Fashion-MNIST dataset**, which contains **60,000** training images and **10,000** test images of various clothing items.  

#### STEP 3: Data Preprocessing  
Convert images to tensors, normalize pixel values, and create **DataLoaders** for batch processing.  

#### STEP 4: Model Architecture  
Design a CNN with **convolutional layers**, **activation functions**, **pooling layers**, and **fully connected layers** to extract features and classify clothing items.  

#### STEP 5: Model Training  
Train the model using a suitable **loss function** (**CrossEntropyLoss**) and **optimizer** (**Adam**) for multiple epochs.  

#### STEP 6: Model Evaluation  
Test the model on unseen data, compute **accuracy**, and analyze results using a **confusion matrix** and **classification report**.  

#### STEP 7: Model Deployment & Visualization  
Save the trained model, visualize predictions, and integrate it into an application if needed.  


## PROGRAM

### Name:AISHWARIYA S
### Register Number:212224240005
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Use torch.nn.functional.relu instead of f.relu
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model
# Train the Model
def train_model(model, train_loader, num_epochs=3):
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

        print('Name: AISHWARIYA S')
        print('Register Number: 212224240005')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch
<img width="230" height="106" alt="image" src="https://github.com/user-attachments/assets/d0673deb-e6cd-4545-b8ea-27667d92f646" />


### Confusion Matrix
<img width="749" height="693" alt="image" src="https://github.com/user-attachments/assets/871fb28c-9734-4a80-a332-302b832d8d81" />



### Classification Report
<img width="492" height="359" alt="image" src="https://github.com/user-attachments/assets/a2536e17-b1e4-4aca-8b0b-9dab92af4e1f" />


### New Sample Data Prediction

<img width="426" height="486" alt="image" src="https://github.com/user-attachments/assets/f6f4e54d-dbec-4f24-970f-a8ce9bd3bd57" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
