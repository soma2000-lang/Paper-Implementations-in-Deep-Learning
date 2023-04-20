# Paper-Implementations-in-Deep-Learning

* Inception-V3 - https://arxiv.org/abs/1512.00567
* Inception-ResNet-V2 and Inception-V4 - https://arxiv.org/abs/1602.07261
* Nasnet  https://arxiv.org/pdf/1707.07012.pdf
* Xception https://arxiv.org/abs/1610.02357

**Inception-ResNet-V2 What I learned**

Inception-ResNet v2 is a deep neural network architecture that combines the Inception module with the ResNet skip-connections to address the issues of vanishing gradients in deep neural networks. This architecture was introduced by Szegedy et al. in 2016 and is considered one of the state-of-the-art models in image recognition and classification.

The Inception-ResNet v2 architecture is a combination of the Inception v4 and ResNet architectures. It has a deep network architecture with 164 layers and has been trained on the ImageNet dataset with over one million image. The architecture uses the Inception module for feature extraction, which is a multi-scale convolutional network with parallel branches of convolutions of different sizes. This module enables the network to extract features at different scales and provides a way to reduce the number of parameters required for the network. also uses residual connections, which help to address the vanishing gradient problem in deep neural networks. These connections allow information to be passed through the network more easily by adding the output of one layer to the input of another. This technique helps to reduce the degradation problem, which occurs when the accuracy of a deep neural network starts to decrease as the network gets deeper.It uses batch normalization, which helps to improve the stability and speed of the training process. Batch normalization involves normalizing the inputs to each layer, which helps to reduce the internal covariate shift that occurs during training.Its trained on the ImageNet dataset, which is a large-scale dataset with over one million images. This dataset includes a wide range of object categories, such as animals, vehicles, and household items. 


***Inception-V3 What I learned***
Inception v3 is a convolutional neural network architecture introduced by Google in 2015, and it is an extension of the original Inception architecture. It has been used for a wide range of computer vision tasks, such as image classification, object detection, and semantic segmentation.
It has a deep neural network structure that consists of many layers, and it uses the Inception module for feature extraction. The Inception module is a multi-scale convolutional network with parallel branches of convolutions of different sizes, which enables the network to extract features at different scales and reduce the number of parameters required for the network.also uses several other techniques to improve its performance. One of the most significant is the use of batch normalization, which helps to improve the stability and speed of the training process. Batch normalization involves normalizing the inputs to each layer, which helps to reduce the internal covariate shift that occurs during training.Another important technique used in the Inception v3 architecture is the use of factorized convolutions, which split the convolution operation into two smaller operations. This technique reduces the number of parameters required for the network and improves the computational efficiency.In addition, the Inception v3 architecture also uses the Global Average Pooling (GAP) layer, which replaces the fully connected layer at the end of the network. The GAP layer takes the average of the feature maps of the last convolutional layer and outputs a single value for each feature map. This technique helps to reduce the number of parameters required for the network and improves the generalization performance.The Inception v3 architecture has been trained on large-scale image datasets such as ImageNet, which is a dataset of over one million images, and achieved state-of-the-art performance on this dataset. It has also been used for object detection and achieved excellent performance on the COCO object detection dataset.


* All models have a consistent pretrained weight loader that adapts last linear if necessary, and from 3 to 1 channel input if desired
### Code
The code here is licensed Apache 2.0. I've taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc. I've made an effort to avoid any GPL / LGPL conflicts. That said, it is your responsibility to ensure you comply with licenses here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue.

### Pretrained Weights
So far all of the pretrained weights available here are pretrained on ImageNet with a select few that have some additional pretraining (see extra note below). ImageNet was released for non-commercial research purposes only (https://image-net.org/download). It's not clear what the implications of that are for the use of pretrained weights from that dataset. Any models I have trained with ImageNet are done for research purposes and one should assume that the original dataset license applies to the weights. It's best to seek legal advice if you intend to use the pretrained weights in a commercial product.

#### Pretrained on more than ImageNet
Several weights included or references here were pretrained with proprietary datasets that I do not have access to. These include the Facebook WSL, SSL, SWSL ResNe(Xt) and the Google Noisy Student EfficientNet models. The Facebook models have an explicit non-commercial license (CC-BY-NC 4.0, https://github.com/facebookresearch/semi-supervised-ImageNet1K-models, https://github.com/facebookresearch/WSL-Images). The Google models do not appear to have any restriction beyond the Apache 2.0 license (and ImageNet concerns). In either case, you should contact Facebook or Google with any questions.
