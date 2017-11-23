# BoilerPlate
Odds and Ends and Things I've implemented.

This is a simple repository containing PyTorch boilerplate code for training classifier nets on various datasets.

The simplest usage is 

```sh
python train.py
```

Which will by default train a WRN40-4 on CIFAR-100, using standard data augmentation, holding out a random (but consistently seeded) 10% of the training set as validation. Take a look at the train.py file for information on the other command line options.

