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

