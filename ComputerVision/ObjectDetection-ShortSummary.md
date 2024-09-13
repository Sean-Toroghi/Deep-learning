<h1>Object detection: short summary</h1>

__References__

- [Book: Computer Vision Projects with PyTorch: Design and Develop Production-Grade Models 2022 ](https://doi.org/10.1007/978-1-4842-8273-1_3)
- [Blog: Using Haar Cascade for Object Detection](https://machinelearningmastery.com/using-haar-cascade-for-object-detection/)
- [Blog: Training a Haar Cascade Object Detector in OpenCV](https://machinelearningmastery.com/training-a-haar-cascade-object-detector-in-opencv/)
- [Book: Computer Vision: Algorithms and Applications, 2nd ed.](https://szeliski.org/Book/)
 

# History

Object detection algorithms could be divided into two categories based on the approach: pre-2012 and post-2012.

The pre-2012 era consists of methods such as HOG, Haar cascades, and variations of SIFT, SURF, and other similar approaches. The post-2012 era consists of algorithms such as RCNN, Fast-RCNN, Faster-RCNN, YOLO (its ever evolving versions), Single Shot Detector (SSD), and more recent algorithms that incorporate transformers and LLM architectures. 

## Boosted Cascade algotihm (pre-2012 era)

The Boosted Cascade algorithm was built initialy to detect faces, but can be used for other detecting tasks. It consists of three sections: 
1. integral images
2. boosting algorithm to select the features
3. a cascade classifier

__Part 1: integral images__

The integral images section gets input images and convert them to integral images, which takes the sum of pixels above and to the left of the point at which pixel values are computed. 

Example of computing integral images, given image in figure 1:
- position 1: pixel sum in rectangule A
- position 2: pixel sum of two rectagles: A + B
- position 3: pixel sum of two rectangle: A + C
- position 4: (pixel sum of position 1 and 4) - (pixel sum of position 2 and 3) = (A + A + B + C + D) - (A + B + A + C) = D

<img src ="https://github.com/user-attachments/assets/64f20210-1eff-4977-b740-4b842a876ea3" width="150" height="150">

__Part 2: boosting algorithm to detect features__

The extracted features from step 1  are plotted against the positive and negative samples and best features are selected.

__Part 3: classifier__

A trained classfier for the positive and negative set of images is form froom the weaker classifier. The _attentional cascade_ in the algorithm helps to reduce computational cost and also improves efficiency of the detector.   The image is divided into multiple sub-windows and for each one, a sequential weak classifier is used to detect a feature. At any step, if a weak classifier fails to detect a feature, the algorithm halts and moves to the next sub-window. The detection succeeds if all classifiers can vote on the presence of the target object and get the bounding box. 

__[Implemention of Boosted Cascade and examples](https://github.com/opencv/opencv/tree/master/data/haarcascades)__

# R-CNN, Fast-RCNN, and Faster-RCNN

## R-CNN

R-CNN algorithm propose an approach for object detection task, in which the algorithm first uses an efficient segmentation algorithm to generate multiple regions. Then similarity scores are calculated acroos all the neighboring elements for all regions. The most similar regions, using a greedy algorithm, are grouped together. The images with region proposals are then processed by convolutional nn for classifying objects (R-CNN uses AlexNet for this task). The extracted features are then evaluated by SVM for classification. Afterall regions are scored, a non-max supression runs on the classified regions and eliminates those regions with IOU less than a threshold value.

The downside of R-CNN is its computational cost. 

## Fast-RCNN

Fast-RCNN solves the issue of R-CNN's computational cost, by introducing pooling operation that reduce the image to a smaller size based on region of interest (ROI). The downside of Fast-RCC is that the process, although is expensive, does not learn any changes in the data. Futhermore, the selective search part of the algorithm is a slow and time-consuming process.

## Faster-RCNN

Faster-RCNN algorithm is an extention of Fast-RCNN that predict the region proposals w/o a selective search method. With a region proposal network, the model identifies the bounding boces in the images and send the same block out to the convolutional neural network to map features. The loss functions are trained on the feature maps. The steps are as following:
- the input image is passed onto a convolutional block to generate the convolutional feature maps.
- A sliding window is used on the feature map for each location, by the region proposal network.
- For each location, nine anchor boxes are used with three different scales and three aspect ratios (1:1, 1:2, 2:1), which helps generate the region proposals.
- The classification layer tells the output whether there is an object present in the anchor boxes.
- The regression layer indicates the coordinates for the anchor boxes.
- The anchor boxes are passed to the region of interest’s pooling layer of the Fast R-CNN architectures.

As shown in the following figure, the model consists of three networks: 
1. head: generates feature maps (ResNet architecutre or similar ones)
2. Region proposal network: generates the region of interest for the classification (opr regression) task
3. Classigication (regression) network: perform the classification task

<img src ="https://github.com/user-attachments/assets/d92e11a8-6bbf-42e2-a835-2b3b4c9370af" width="300" height="350"> [Ref.](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781484282731/files/images/520381_1_En_3_Chapter/520381_1_En_3_Fig7_HTML.jpg)

__Faster R-CNN layers can be divided into four main parts: Anchor generation layers -> region proposal layers -> ROI pooling layers -> classification layers__

- Anchor generation layers: produces a series of content-agnostic bounding boxes with different sizes and aspect ratios to cover most of the image regions. A differnt approach is to define a refference boudning box and try to predict and correct the offset values to make a better fit.
- Region proposal layers: changes the position, width, and height of the anchor poxes to fit the object better. It consists of four sections: regional proposal network, proposal layer, anchor target layer, and proposal target layer. 
- ROI pooling layers: gets the images to fixed dimensions for the last layers.
- classification (regression) layers: performs classification (regression).


## Mask-RCNN

An extention to Faster-RCNN is mask-RCNN, which predicts masks on the detected objects. It has two morel ayers after the ROI pooling layer for adding masks. Using ROI align, it  aligns the extracted features with the inputs. It also uses bilinear interpolation to get the exact or near-perfect values of the input regions.



# YOLO
