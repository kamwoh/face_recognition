# Face Recognition based on DeepID

Implementation of DeepID based on the paper "Sun Y, Wang X, Tang X. Deep learning face representation from predicting 10,000 classes[C]//Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014: 1891-1898."

## Dataset preparation
LFW - refer to sklearn.dataset

Facescrub - http://vintage.winklerbros.net/facescrub.html

Cropped only faces, separate them into train, val, and test set with ratio of 0.7, 0.1, 0.2 respectively

## Current state
Only done face identification, working on face verification

## Training
Initially learning rate of 0.01 using exponential decay on 100000 steps/0.9 decay rate

Monitor the training graph, if it stays at a loss/accuracy for a long time, initialise learning rate with 0.005 or lower (my guess on it, i think it is because it reaches a local minimum gradient, couldn't go deeper)

## Reminder
1. Small dataset will be easily overfit as there is nothing much to "learn" from the dataset

2. Due to Internet speed and storage problem, I choose a smaller than CASIA dataset (stated in the paper), but bigger than LFW which is facescrub

3. My code is in continue training state, if you want a new training, comment the "load" code

## Performance
Training on LFW - maximum of 80% accuracy (only 68 classes, I choose minimum of 10 faces)

Training on Facescrub - still training, but reached 75% accuracy by the time I commit (530 classes)

## Contact

Email: kamwoh@gmail.com

## Reference

[1]. https://github.com/RiweiChen/DeepFace

[2]. https://github.com/stdcoutzyx/DeepID_FaceClassify

[3]. Sun Y, Wang X, Tang X. Deep learning face representation from predicting 10,000 classes[C]//Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014: 1891-1898.