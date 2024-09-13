<h1>Object detection: short summary</h1>

# History

Object detection algorithms could be divided into two categories based on the approach: pre-2012 and post-2012.

The pre-2012 era consists of methods such as HOG, Haar cascades, and variations of SIFT, SURF, and other similar approaches. The post-2012 era consists of algorithms such as RCNN, Fast-RCNN, Faster-RCNN, YOLO (its ever evolving versions), Single Shot Detector (SSD), and more recent algorithms that incorporate transformers and LLM architectures. 

## Boosted Cascade algotihm (pre-2012 era)

The Boosted Cascade algorithm was built initialy to detect faces, but can be used for other detecting tasks. It consists of three sections: 
1. integral images
2. boosting algorithm to select the features
3. a cascade classifier

### Part 1: integral images

The integral images section gets input images and convert them to integral images, which takes the sum of pixels above and to the left of the point at which pixel values are computed. 

Example of computing integral images, given image in figure 1:
- position 1: pixel sum in rectangule A
- position 2: pixel sum of two rectagles: A + B
- position 3: pixel sum of two rectangle: A + C
- position 4: (pixel sum of position 1 and 4) - (pixel sum of position 2 and 3)

<img src ="https://github.com/user-attachments/assets/64f20210-1eff-4977-b740-4b842a876ea3" width="150" height="150">



