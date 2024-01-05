# The Simpsons Characters classification. Kaggle competition.  

In this project I will solve the competition from [kaggle](https://www.kaggle.com/competitions/journey-springfield). The main task will be to train a classifier based on convolutional networks to learn how to distinguish all the residents of Springfield. 

$\textbf{Description:}$

Today you have to help FOX television companies process their content. As you know, the Simpsons series has been on television for more than 25 years, and during this time a lot of video material has been copied. Characters have changed as graphics technology has changed, and 2018 Homer Simpson doesn't look much like 1989 Homer Simpson. In this task you need to classify the characters living in Springfield. I think there is no point in supporting each of them separately.

<img src='https://assets-prd.ignimgs.com/2023/09/25/simpsons-ver51-xlg-button-1695660401542.jpg' width=500>

$\textbf{Lineup of this project is:}$

1. Baseline solution -> simple CNN
2. More epochs and hyperparameters tunning
3. Picture normalization & Batch Size
4. Different Poolings & BatchNorm
5. Testing different optimizers
6. Transfer Learning. ResNet and Inception fine-tunning
7. More pictures -> better quality. Trying some tricks

$\textbf{Evaluation:}$

The metric for this competition is the Mean F1-Score. The F1-measure is calculated based on the precision P and recall R. Accuracy is the ratio of true positives (TP) to all predicted positives (TP + FP). Completeness is the ratio of true positives to all actual positives (TP + FN). Then F1 is given by the formula:

$$F_1 = 2 \frac{P \cdot R}{P + R}, \text{ where } P = \frac{TP}{TP + FP} \text{ and } R = \frac{TP}{TP + FN}$$

## Baseline Solution 

After preprocessing the input dataset (which includes pictures -> tensors stage and input normalizatio, in a common way, with mean and std form ImageNet) I implement the simple CNN architecture, which I will be used for baseline solution.  

I used CNN with the following structure: 

<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Simpsons%20Classification/images/nn_baseline.svg' width=3000>

Clearly, the structure is : 

```
SimpleCnn(
  (conv1): Sequential(
    (0): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv4): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv5): Sequential(
    (0): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (out): Linear(in_features=2400, out_features=42, bias=True)
)
```

We have 5 convolution layers and predict 42 classes (number of characters in Simpson's dataset). Also I used: 

| Optimizer | Loss |
| ------- | --- |
| Adam | CrossEntropyLoss | 

The learning curve looks like this: 

<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Simpsons%20Classification/images/loss_plot_baseline_cnn.png' width=1000>

We can see that about 3 epochs is enough for this cnn architecture to reach maximum accuracy on validation data. 

After sending the baseline predictions to kaggle we can see the final score. 

| Model | Public Score on Kaggle |
| ------- | --- |
| Simple CNN | 0.90648 | 
