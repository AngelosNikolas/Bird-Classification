# 250 species of birds classification Using the VGG16 Pre-trained Model

## Angelos Nikolas

Using the VGG16 Pre-trained model the classification of 250 species of birds is explored. The model is fitted for a limited time and providing almost 90% accuracy, the accuracy could be higher if the training time was increased. Additionaly, a function for classifying any bird is given.

### Code Overview
For the bird classification the data are loaded, then ImageDataGenerator is used so the model is receiving variations of the images and only returns the transformed images preventing overfittings for all sets created. Transferred learning was utilized by deploying the VGG16 pre-trained model. The weights are set with the standard Imagenet, the input shape is set at 130. Next the all the layers are set to freeze meaning at this state they won’t be update during training this is reverted before the fine-tuning phase. The output layer is set 250 for the classification of the 250 bird classes. Adam optimizer is used due to versatility and good performance on this kind of networks. After unfreezing the base model fine-tuning occurs to jointly train both the newly added classifier layers and the last layers of the base model. 5 epochs were used for training the base model and additional 5 epochs for the fine-tuning. 
### Scores 
Accuracy 0.8912, precision 0.7999, recall 0.7991
