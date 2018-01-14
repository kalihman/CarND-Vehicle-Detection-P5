## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Notes

* vehicle_detector.py contains all the necessary class, functions, preprocessor, pipeline, and main scripts to process the video.

* The final output video, with cars detected, is project_output.mp4.

* To run this script, you will need to donwload vehicle and non-vehidle image files in the same directory of vehicle_detector.py script


[//]: # (Image References)
[car]: ./output_images/1.png
[notcar]: ./output_images/image7.png
[hog_output]: ./output_images/hog_output.png
[windows96]: ./output_images/windows96.png
[cars64]: ./output_images/cars64.png
[cars160]: ./output_images/cars160.png
[heatmaps]: ./output_images/heatmaps.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The function that extracts HOG features is called `get_hog_features`, at [line 66](vehicle_detector.py#L66). During the training step, HOG features are extracted within the `extract_features` function, which is defined at [line 109](vehicle_detector.py#L109)  and called at [lines 409](vehicle_detector.py#L409) and [418](vehicle_detector.py#L418). During the processing pipeline, HOG features are extracted within `single_img_features`, which is defined at [line 223](vehicle_detector.py#L223) and called by the `search_windows` function, which in turn is called at [line 276](vehicle_detector.py#L276).

I also added the option to include color features (a coarsened and flattened representation of the image) as well as histograms of the intensities of each color channel present in the image.  `bin_spatial`, at [line 88](vehicle_detector.py#L88), computes the color features, and `color_hist`, at [line 97](vehicle_detector.py#L97), computes the histogram of color intensities.  Both `bin_spatial` and `color_hist` are also called from `extract_features` and `single_img_features`.

The training set contains of several thousand 64x64 images of cars of various models and colors, taken 
from different angles.  Here is an example:

![car from training set][car]

The training set also contains several thousand "non-car" images, which are 64x64 pictures of "non-car"
features such as lane lines, trees, empty roadway, signs, etc. that are typically encountered while driving.  
Here is an example of a "non-car" image:

![not car from training set][notcar]

My feature extraction functions included options to explore different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). This proved useful when constructing an image processing pipeline later (I was able to choose the best parameters by trial and error).

Here is a visualization of a HOG for the above car image, using color channel G in RGB color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog output][hog_output]

#### 2. Explain how you settled on your final choice of HOG parameters.

I had already tried various combinations of HOG parameters when using the HOG in the quizzes, based on how accurately an SVM trained using those HOG parameters performed on a test set of data. I transferred those parameters over and they performed well.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using color features and HOG features extracted from the images in the "vehicles" and "non-vehicles" directories. Training data consisted of 8000 vehicle images and 8000 non-vehicle images.  Images were randomly shuffled prior to training. I split off 20% of the training data to use as a test set, so the SVM was actually only trained on 6400 vehicle images and 6400 non-vehicle images.

I tried training with various combinations of color features, color-histogram features, and HOG features. I also experimented with different color spaces, as well as different internal parameters associated with each set of features (like number of spatial bins for color features, number of histogram bins for color histograms, and HOG parameters described above).

I found that when I used all three feature types, accuracy on the test set was highest (>99%). However, including color histograms as a feature seemed to make it more difficult for the pipeline to detect the black car. Therefore, I used only color features and HOG features. Although omitting color histograms reduced the accuracy of the SVC on the test set to around 98%, it caused the SVC to recognize the black and white cars equally well. The final parameters I used can be found at [line 382-391](vehicle_detector.py#L382-391). 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched using four different image scales (64x64, 96x96, 128x128, and 160x160 pixels), which seemed reasonable based on the sizes nearby, mid-distance, and faraway cars present in the test images and videos. In all cases, I scanned the images across the chosen y-region of interest with an overlap fraction of 0.5, which proved sufficient to enable reliable detections. y-regions of interest are defined at [line 493](vehicle_detector.py#L493). The function to create the set of windows to process for each scale is `slide_window`, defined at [line 170](vehicle_detector.py#L170).  Within my processing pipeline, `slide_window` is called at [line 507](vehicle_detector.py#L507).


Here is an example of the set of windows produced for a window scale of 96x96:
![Windows used for the 96x96 scale][windows96]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I optimized my classifier as described in "3. Describe how (and identify where in your code) you trained a classifier..." above.

As an example of the performance of different window sizes, here is a test image showing car regions identified by the 64x64 sliding windows:

![Car regions identified by 64x64 windows][cars64]

Here is that same image showing the car regions identified by 160x160 sliding windows:

![Car identified by 160x160 windows][cars160]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The final output video is [project_output.mp4](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each frame, I created a heatmap by adding 1s to the regions of a blank grayscale image corresponding to window (of all 4 scales)  identified as containing cars. I added the heatmap for the current frame to a queue of the most recent several frames, and created a total heatmap consisting of the sum of the heatmaps for over the queue. In the summed heatmap, intensity values below a certain threshold were zeroed. This process of summing over a brief history and then thresholding helped remove transient false positives and strengthen persistent detections. The number of frames to keep in the history queue (`n_history`) and the threshold value were tunable parameters; I ended up using a history of 5 frames and a threshold value of 7.

The history queue and the running total heatmap were tracked by an instance of a RunningHeatmap class, defined at [line 22](vehicle_detector.py#L22). Its use in my processing pipeline appears at [lines 469](vehicle_detector.py#L469) and [537](vehicle_detector.py#L537).

I then applied `scipy.ndimage.measurements.label()` to identify individual island regions of detection in the history-summed heatmap. I assumed each blob corresponded to a vehicle, and constructed bounding boxes to cover the area of each island region. These bounding boxes were then drawn onto the output image in my processing pipeline ([line 556](vehicle_detector.py#L556)).

I experimented with a small range of values for the number of history frames to keep, and the threshold below which the history-summed heatmap was set to zero.  I ended up using a history of 5 frames and a threshold of 7.

**Top left:**  example of a heatmap from an individual frame of test_video.mp4.
**Top right:**  the summed heatmap from that frame and the four previous frames.  
**Bottom left:**  output of `scipy.ndimage.measurements.label()` on the history-summed and thresholded heatmap.  
**Bottom right:**  bounding boxes of the labeled regions drawn onto the current frame.  
![Heatmaps and resulting detections][heatmaps]  
A pair of spurious detections can be seen in the instantaneous heatmap (top left) but these have a much smaller relative intensity in the history-summed heatmap (top right) and are eliminated by thresholding. Therefore, the false-detection regions are not labeled as car regions or drawn on the output frame.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the parameters I used had been chosen in earlier experimentation in the quizzes, with the goal of optimizing the performance of the linear SVC on the test set of my data. Trying different parameters to get good performance on the test set was a relatively quick process, so achieving high accuracy on the test set was not difficult.  The main difficulty arose when my SVC had trouble recognizing the black car in the test video. I eventually discovered that omitting the color histogram from the features allowed the SVC to recognize the black car more consistently, even though it reduced the SVC's accuracy on the test set.

My pipeline seems robust overall, at least on the videos provided. In the final output, all cars are tracked, and false detections are minimal. However, the pipeline does briefly lose track of the white car at a certain distance range, then find it again (see 0:24-0:26 of project_output.mp4). This could be because there is a "sour spot" in the scales of sliding windows I chose to use, 
where the car may be too small to be identified by the 96x96 windows but too large to be identified by the 64x64 windows.  I could probably improve this by searching a finer range of window scales with a greater overlap fraction.

The biggest weakness of my pipeline is its relatively high computational cost. On my PC, it took about 40 minutes to process the project video. For simplicity, I initially wrote the pipeline such that after selecting each search window, it samples to 64x64, then creates a new HOG. A more efficient alternative would be to compute the HOG once for an entire image, then for each search window, take the preexisting HOG and interpolate it to a size corresponding to a 64x64 image, as described in the lessons.  If I had to do futher experimentation with the final project video, I would implement this method.

