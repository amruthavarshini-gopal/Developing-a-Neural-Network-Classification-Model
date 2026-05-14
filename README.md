# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the customer segmentation dataset using Pandas and preprocess the data by removing unnecessary columns, handling missing values, and encoding categorical features.

### STEP 2: 
Split the dataset into input features (X) and target labels (y), then divide the data into training and testing sets using train_test_split().

### STEP 3: 
Create a StandardScaler object, fit it with the training data, and normalize both training and testing feature sets.

### STEP 4: 
Convert the processed data into PyTorch tensors and create TensorDataset and DataLoader objects for batch processing.

### STEP 5: 
Build the Neural Network model using PyTorch by defining input, hidden, and output layers with ReLU activation functions.

### STEP 6: 
Define the loss function (CrossEntropyLoss) and optimizer (Adam), then train the neural network model using the training data for multiple epochs.

### STEP 7:
Evaluate the trained model using the testing data and calculate performance metrics such as accuracy, confusion matrix, and classification report.

### STEP 8:
Plot the confusion matrix using Seaborn and Matplotlib to visualize the classification performance.

### STEP 9:
Use the trained neural network model to predict the customer segmentation class for a sample test input and compare the predicted value with the actual value.

## PROGRAM

### Name: Amruthavarshini Gopal

### Register Number: 212223230013

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)

    def forward(self, x):
        #Include your code here
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here
    model.train()
    for epoch in range(epochs):
      for inputs,labels in train_loader:
        optimizer.zero_grad();
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

   if (epoch + 1) % 10 == 0:
         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model =PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)

```

### Dataset Information
<img width="1520" height="281" alt="Screenshot 2026-05-14 154629" src="https://github.com/user-attachments/assets/442ca30c-58fb-4b90-a11e-24e119bf8bbf" />


### OUTPUT

## Confusion Matrix

<img width="786" height="632" alt="Screenshot 2026-05-14 154803" src="https://github.com/user-attachments/assets/d68fdd58-2bf3-44f7-8b25-c0f66184a6eb" />


## Classification Report
<img width="757" height="499" alt="Screenshot 2026-05-14 154834" src="https://github.com/user-attachments/assets/b7a9d433-ccce-412d-a2d1-f841312f92e7" />


### New Sample Data Prediction
<img width="563" height="119" alt="Screenshot 2026-05-14 154923" src="https://github.com/user-attachments/assets/ff67c6bc-1b95-4685-84d1-53da1e0f36f0" />

## RESULT
Thus, a Neural Network Classification model was successfully developed and trained using PyTorch for customer segmentation prediction.
