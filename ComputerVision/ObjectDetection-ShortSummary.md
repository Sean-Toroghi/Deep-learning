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

__[Implementiion of Boosted Cascade and examples](https://github.com/opencv/opencv/tree/master/data/haarcascades)__

