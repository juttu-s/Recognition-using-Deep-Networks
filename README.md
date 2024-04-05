# README - Project 5
Name - **SAIKIRAN JUTTU** & **KIRTI KSHIRSAGAR**

- Using Windows 10, VScode, OpenCV 4.9(recent one), Pytorch and python.
- Using 3 Time Travel Days (submitting on 4th April 2024)

## This is a Readme file for Project-5, which includes all the details that are necessary to run the program. For this project, we have differnt files for different tasks.

### Task-1
- This task has been implemented using 4 different files. Tasks from **A-D** are implemented in **Task1ABCD.py** file. Task **E** and **F** are implemented in **Task1E.py** and **Task1F.py** respectively.
- In this task, we build and train a network to do digit recognition using the MNIST data base and save the network to a file so it can be re-used for the later tasks - **model.pth** .

    - Part A: In this task, we import the dataset and use matplotlib pyplot to look at the first six example digits of the test set.
    - Part B: In this task, we build a network model with the given layers.
    - Part C: In this task, we train the model for **10 epochs** and evaluate the model on both the train and test sets after each epoch. We chose the batch size to be **64**. We also generate plots for train and test accuracy.
    - Part D: In this task, after training the network we saved it to a file in the results folder.
    - Part E: In this task, we read the network in evaluation mode and run the model on the first 10 examples in the test set. 
    - Part F: In this task, we have created a folder containing handwritten digits and processed the images and matched their intensities with the intensities of the test data. We have then evaluated the model on these handwritten digits and checked how well it performs.

### Task-2
- This task has been implemented in the **Task2.py** file.
- In this task, we examine our network and analyze how it processes the data.
    - Part A: In this task, we analyze the first layer and visualize the filters present in that layer.
    - Part B: In this sub-task, we show the effect of the filters on image from the dataset.

### Task-3
- This task has been implemented in the **Task3.py** file.
- In this task, we perform Transfer Learning on Greek Letters. We used the previously developed MNIST digit recognition network to identify three specific Greek letters: alpha, beta, and gamma.
- Here, we loaded pre-trained weights from a file, froze the network weights and replaced the last layer with a new Linear layer with three nodes.
- We also investigated the number of epochs required to achieve nearly perfect identification using the 27 provided examples.
- We also visualized a plot depicting the training error.
- Later, we used our images of alpha, beta and gamma to evaluate the classification accuracy of the trained network. 

### Task-4
- This task has been implemented in the **Task4.py** file.
- In this task, we evaluated the effect of different network architecture parameters on the performance and training time of a deep neural network for the **MNIST Fashion** dataset classification task.
- The dimensions which we considered for our experiments are: 
    - The size of the convolution filters
    - The dropout rates of the Dropout layer
    - Whether to add another dropout layer after the fully connected layer
    - The number of epochs of training
    - The batch size while training


# Extension:
For the extensions, we have done the following tasks:
- Task 1 : This extension has been implemented in file called **Extension1.py**. For the first extension, we tried loading one pre-trained network (**ResNet18**) and evaluated its first couple of convolutional layers as we did in task 2.
- Task 2 : This extension has been implemented in file called **Extension2.py**. In this extension, we replaced the first layer of the MNIST network with **Gabor filters** and retrained the rest of the network, holding the first layer constant. We then compared the results from this and the original network.
- Task 3 : This extension has been implemented in file called **Extension3.py**. In this extension, we build a live video digit recognition application using the trained network. The saved trained model and its optimizer were loaded. We then initialized the webcam to capture video frames in real-time. Each frame underwent preprocessing to ensure compatibility with the neural network's input format. The live video stream was displayed alongside the predicted digit, providing real-time digit recognition.
- Task 4 : We Evaluated more dimensions on task 4. Further details are mentioned in task 4.

