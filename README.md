# Yu solution for Diabetic Retinopathy competition

This is a competition in Kaggle [Diabetic Retinopathy](https://www.kaggle.com/c/diabetic-retinopathy-detection), where your task is to classify each person's eye examination to 5 different degree of disease, resulted from diabetic.

This is repository for the codes I used to process the original image, the convolution neural net model(build with keras). It is mainly based on one benchmark provided in the forum. With the following steps:

- Only used the normal image processing to 256X256, didn't use other way to adjust color etc.

- Balance different classes' pictures by augmenting class 1, 2, 3 and 4.

- Used VGG style architecture, trained with 10 epoch, batch size 32, with Kepler K20c GPU. It runs for about 2 days. 

- Instead of running as classification problem, since the output are ordered(stage of disease), we run as regression problem instead.

- Convert the raw output to disease stage labels. Naively, we can convert to the closely stage. But ranked the raw scores based on the original dataset's proportion yields a better Kappa score. 

In the end the model yields Kappa of 0.38 on private dataset, which is close the 0.387 score on the public score due to the dropout, and quite large a sample size. 

The model is by no means useful(the winners reach 0.85), which is a very impressive and potentially very applicable model. 

This is just an experiment and hands on experience with convolutional neural network, after following [Standford CS231n](http://cs231n.stanford.edu/index.html) course.

For references, there are several shared solution on the forum after the competition was over. They can be found [Jeffery](http://jeffreydf.github.io/diabetic-retinopathy-detection/), [Ilya Kavalerov](http://ilyakava.tumblr.com/post/125230881527/my-1st-kaggle-convnet-getting-to-3rd-percentile) and [Mathis and Stephan](https://www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/15617/team-o-o-solution-summary). A lot of ideas and inspiration! Until next time.

