# Motivation of The Proposed Methods

Balanced dataset is always the desired dataset for training machine learning models.
However, one can never get data that is equally distributed samples among all classes in
nature. Imbalanced classification is a crucial problem that data scientists have to deal with,
since it leads ML models to have poor predictive performance, specifically for the minority
classes. This report discusses a method, the re-weighted random sampler combined with
data augmentation that improves convolutional neural networks’ performance on imbalanced
datasets.

# Problem Formulations and Data Collections

This report discusses the re-weighted random sampler combined with data augmentation’
effect on imbalanced classification. This method can be splitted into 2 parts. The first one is
the re-weighted random sampler.

# Re-weighted Random Sampler

In imbalanced classification, since there are more samples in majority classes than minority
classes, the ML(machine learning) model will learn more samples from the majority classes’
samples. Such bias results in high accuracy in majority classes but low accuracy in minority
classes. Therefore, the target of the re-weighted sampler is to let the ML model learn the
same number of samples from each class. In order to develop a formula that calculates the
weights for each dataset, the details of each dataset need to be known.

The cifar_50 balanced and long tailed datasets with imbalanced ratio = {0.1, 0.02, 0.005} are
used to test the performance of the proposed method [1]. The shapes of the long tailed
cifar_50 datasets are shown in fig 1, fig.2 and fig.3. The CT scans of breast cancer dataset
is used as a supplemental dataset [2].

![image](https://user-images.githubusercontent.com/58502695/138342169-a043dbb0-0b71-492f-abef-c4c79a9d5b0d.png)
![image](https://user-images.githubusercontent.com/58502695/138342209-242c6322-e204-4f9b-a290-afcdd3f175b7.png)
![image](https://user-images.githubusercontent.com/58502695/138342235-35938480-1a70-4fde-a319-58655f1de38f.png)

Long-tailed versions of CIFAR-50 are created by reducing the number of training samples per class according to an exponential function shown in Fig.4 [3]. 

![image](https://user-images.githubusercontent.com/58502695/138342324-eb03f0c4-fdaa-43c3-bc11-238030067c3c.png)

“n” is the number of samples, “u” is the imbalanced ratio = {0.1, 0.02, 0.005} and “x” is the class ID. To make the ML model learn the same number of samples from each class, class_weight x number of samples should be the same for each class. I developed a formula to calculate weights shown in Fig. 5. The calculated weights are shown in Fig. 6. The first row is the number of samples in each class, the second row is the weights and the columns are the different long-tailed datasets.

![image](https://user-images.githubusercontent.com/58502695/138342382-fdee868f-3cc6-454a-9cf1-9402678235fb.png)

![image](https://user-images.githubusercontent.com/58502695/138342411-92cd1bd0-5fb3-47ba-a380-77b48a0af869.png)


# Data Augmentation

If re-weighted sampler is the only method implemented, the re-weighted sampler will make the model learn a few samples many times in tailed classes and thus the ML model will be overfitted on the few samples in the tailed classes. Therefore, data augmentation needs to be implemented to generate new samples [4]. The transform list is in Fig.7. Everytime the iterator takes a picture from the dataloader, the picture has 30% chance to be rotated, 30% chance to have affine transform, 30% chance to be flipped horizontally and 30% chance to be passed in a color jitter filter. These chances and transforms are not exclusive. 

![image](https://user-images.githubusercontent.com/58502695/138342534-d87528bf-8000-426a-a5fb-59cdc83333d3.png)

# Realistic Constraints

The runtime of this experiment greatly depends on the device. The GeForce RTX 3070 Laptop GPU, shown in Fig. 8, is used in this experiment. For training 50 epochs in total, it took 4 minutes to store the 25000 images of cifar-50 balanced dataset to GPU memory the first time, and the following 49 epochs took 35 minutes. The runtime varies greatly on different devices. I tried to train a cifar-50 balanced dataset on Google colab GPU, which has 2496 CUDA cores , 12GB GDDR5 VRAM[5].  The first epoch took 3.5 hours and the rest 49 epochs took 2 hours 5 minutes. 

![image](https://user-images.githubusercontent.com/58502695/138342638-25f3af76-0eb5-44d5-9ec2-dcf94c464c3c.png)

Different machine learning models influence the training time. As the model complexity (layers and neurons) increases, the runtime increases significantly. The model structure that I am using is shown in Fig. 9. 


