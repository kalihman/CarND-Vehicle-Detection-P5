from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split

############
# 1. Class #
############

# running_heatmap is a class that stores a queue of the detection heatmaps
# from the last n_history frames, as well as a total heatmap which is computed
# as a running total of the last n_history detection heatmaps.
class RunningHeatmap:
    def __init__(self, n_history=3):
        # The most recent n_history heatmaps are stored in self.queue.
        # self.queue[0] will be the oldest, and self.queue[n_history-1] will
        # be the newest.
        self.queue = []
        self.initialized = False
        self.n_history = n_history

    # Member function to add the heatmap from the next frame.
    def push(self, new_heatmap):
        # The first time we call push(), we need to initialize the history queue.
        if self.initialized == False:
            # Since we only have a "history" of one frame at the moment, we initialize 
            # the history queue with n_history copies of the first heatmap.
            for i in range(0, self.n_history):
              self.queue.append(new_heatmap)
            self.initialized = True
        else:
            # Current heatmaps in the queue are pushed down towards lower indices.
            # The heatmap at the self.queue[0] is lost.
            for i in range(0, self.n_history-1):
                self.queue[i] = self.queue[i+1]
            # The new heatmap is inserted at self.queue[n_history-1]
            self.queue[self.n_history-1] = new_heatmap
        # Recompute the running heatmap from the new history queue.
        self.compute_running_heatmap()

    def get_running_heatmap(self):
        return self.running_heatmap

    # Member function to compute the current running-total
    # heatmap by summing the heatmaps currently in the history queue.
    def compute_running_heatmap(self):
        self.running_heatmap = np.zeros_like(self.queue[0])
        for i in range(0, self.n_history):
            self.running_heatmap += self.queue[i]

#######################
# 2. Helper functions #
#######################

# This function is based on code from the quizzes.
# It returns a histogram of oriented gradients for an input image.
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# This function is based on code from the quizzes.
# It gives a feature vector consisting of coarsened spatial features of an
# input image.
def bin_spatial(img, size=(32, 32)):
    # Coarsen the input image to the desired resolution, then flatten it
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# This function is based on code from the quizzes.
# It computes a histogram of the intensities for each color channel,
# then concatenates those histograms into a feature vector.
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Create the feature vector by concatenating the three histograms
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# This function is based on code from the quizzes.
# It is used to read the training data from a list of image filenames.
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, is_png=True):
    # Create a list of feature vectors
    features = []
    # Iterate through the list of image filenamese
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        if( is_png ):
           image = (255*image).astype(np.uint8) 
        # print(np.max(image))
        # plt.imshow(image)
        # plt.show()
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            # Create spatial features by coarsening the input image, then flattening it
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Create a color histogram of the training image as another set of features
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # If 'ALL', create a histogram of oriented gradients for all color channels
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                # If a specific color channel was selected, create a HOG only for that channel
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# This function is based on code from the quizzes.  
# It takes an x range, a y range, a window size, and a window overlap
# fraction, and returns a list of (overlapping) windows that step across
# and fill the specified x-y region.
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to entire image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# This function is based on code from the quizzes.
# It takes a list of boxes and draws them on an input image.
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# This function is based on code from the quizzes.
# It extracts features from an image in a manner similar to extract_features,
# but unlike extract_features, it only operates on one image instead of a list.
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
# just for a single image rather than list of images
# just for a single image rather than list of images
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# This function is based on code from the quizzes.
# It takes a list of boxes within an image to search (the output of slide_window above).
# For each box, it rescales the contained subimage to 64x64, extracts features,
# and runs it through the classifier to see if the classifier detects a car 
# within that box.
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    # Create an empty list to receive positive detection windows
    on_windows = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # Scale extracted features to be fed to classifier,
        # using the scaler that was fitted to the training data earlier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using given classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# This function is based on code from the quizzes.
# This function takes a list of boxes in which a car was detected
# For each box, it increments a heatmap by 1 for those regions
# contained by the box.
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

# This function is based on code from the quizzes.
# It takes an image and a labels object.
# labels[1] contains the number of unique hot regions (corresponding 
# to the presence of a car) found, and labels[0] contains a grayscale
# map of the hot regions, where each unique region is assigned a unique
# grayscale value indexed from 1 to labels[1].
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
    # Return the image
    return img

#################
# 3. Preprocess #
#################

# Read in cars and notcars datasets
images_vehicles = []
images_vehicles.append(glob.glob('vehicles/GTI_Far/*.png'))
images_vehicles.append(glob.glob('vehicles/GTI_Left/*.png'))
images_vehicles.append(glob.glob('vehicles/GTI_MiddleClose/*.png'))
images_vehicles.append(glob.glob('vehicles/GTI_Right/*.png'))
images_vehicles.append(glob.glob('vehicles/KITTI_extracted/*.png'))

images_nonvehicles = []
images_nonvehicles.append(glob.glob('non-vehicles/Extras/*.png'))
images_nonvehicles.append(glob.glob('non-vehicles/GTI/*.png'))

cars = []
notcars = []
for imagelist in images_vehicles:
    for image in imagelist:
        cars.append(image)
for imagelist in images_nonvehicles:
    for image in imagelist:
        notcars.append(image)

# Shuffle the training data to reduce the possibility of choosing a 
# time-series of very similar images when selecting a subset of the data
shuffle(cars)
shuffle(notcars)

# Select a subset of the data to train on (if necessary)
# This can reduce overfitting.
# I also take this opportunity to ensure that the "car"
# and "not car" sets are the same size, which is beneficial
# when training a support vector machine.
sample_size = 8000
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

# Set hyperparameters to use when extracting features
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # Number of orientation bins to use for histogram of oriented gradients (HOG)
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off

# The following two hyperparameters are used to remove transient
# false positives from the heatmap of pixels identified as containing 
# a car.
# Number of heatmaps from previous frames that are summed
# to create the current total heatmap
# e.g., if n_history = 3, the current frame's heatmap and 
# the heatmap from the 2 previous frames are summed,
# and the summed heatmap is then thresholded and used to label cars.
n_history = 5

# Threshold below which the history-summed heatmap should 
# be zeroed to reduce false detections
threshold = 7

# Extract feature vectors from training images labelled as cars.
# (see functions.py)
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Extract feature vectors from training data labelled as "not cars."
# (see functions.py)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

# Fit a per-column (per-feature) scaler that will
# ensure each feature has zero average and unit standard deviation
X_scaler = StandardScaler().fit(X)

# Apply the scaler to the features from the training data
scaled_X = X_scaler.transform(X)

# Create the labels vector from the training data
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear support vector classifier, as in the quizzes
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the accuracy using the test set
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single test image
t=time.time()

# Declare a running heatmap object (see functions.py)
# to store the heatmaps from the last n_history frames
# and sum them to create the total heatmap used to 
# identify car locations. This moving sum helps reduce 
# the influence of transient false positives.
# I declare heat_running as a global object because it must be used persistently
# across invocations of the "pipeline" function below.
heat_running = RunningHeatmap(n_history)

# nframes tracks the total number of frames we have processed so far.
nframes = 0

###############
# 4. Pipeline #
###############

# "pipeline" is the function that will be used to pass each  
def pipeline(input_image):
    # nframes and heat_running are both persistent 
    # and modified between invocations of "pipeline."
    global nframes
    global heat_running
    
    print("Processing frame {}".format(nframes))
    nframes += 1
    draw_image = np.copy(input_image)
   
    # Number of different window sizes to try during sliding window search 
    window_sizes = [64,96,128,160]

    # y-region to search for each choice of window size
    y_start_stops =[[np.int(input_image.shape[0]*6/11), input_image.shape[0]*9/11], 
    	            [np.int(input_image.shape[0]*6/11 ), input_image.shape[0]], 
    	            [np.int(input_image.shape[0]*6/11 ), input_image.shape[0]],
    	            [np.int(input_image.shape[0]*6/11 ), input_image.shape[0]]]

    # heatmap for this frame, created as a sum of the heatmaps 
    # for each choice of window size
    heatmap = np.zeros_like(input_image[:,:,0]).astype(np.uint8)

    # Loop over window sizes
    for window_size,y_start_stop in zip( window_sizes, y_start_stops ):

        # Create the list of windows locations to be searched for this window size
        # (see functions.py)
        windows = slide_window(input_image, 
                               x_start_stop=[None, None], 
                               y_start_stop=y_start_stop, 
        	                   xy_window=(window_size, window_size), 
                               xy_overlap=(0.5, 0.5))

        # Find the list of windows where a car is detected        
        # (see functions.py)
        hot_windows = search_windows(input_image, windows, svc, X_scaler, 
                                     color_space=color_space, 
                                     spatial_size=spatial_size, 
                                     hist_bins=hist_bins, 
                                     orient=orient, 
                                     pix_per_cell=pix_per_cell, 
                                     cell_per_block=cell_per_block, 
                                     hog_channel=hog_channel, 
                                     spatial_feat=spatial_feat, 
                                     hist_feat=hist_feat, 
                                     hog_feat=hog_feat)                       
        
        # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=3) 
        # hot_window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=3)                    

        # Add heatmap for this window size to the heatmap for this frame
        add_heat(heatmap, hot_windows)

    # Add the heatmap for this frame to the heat_running object's queue of
    # the last n_history frames.  This call also forces heat_running
    # to update its internal total heatmap (the sum of heatmaps for the
    # last n_history frames)
    heat_running.push(heatmap)

    # Get a copy of the current total heatmap (sum of last n_history heatmaps)
    thresholded_total_heatmap = np.copy(heat_running.get_running_heatmap())
    # Threshold the total heatmap to reduce false detections
    thresholded_total_heatmap[thresholded_total_heatmap <= threshold] = 0
 
    # Label each unique island region of the thresholded heatmap
    # as a separate car
    labels = label(thresholded_total_heatmap) 

    # Draw the bounding box of each labelled island region
    # on the final output image
    tracking_image = draw_labeled_bboxes(draw_image, labels)

    # fig = plt.figure()
    # plt.subplot(221)
    # plt.imshow(heatmap, cmap='hot')
    # plt.subplot(222)
    # plt.imshow( heat_running.get_running_heatmap(), cmap='hot' )
    # plt.subplot(223)
    # plt.imshow(labels[0], cmap='gray')
    # plt.subplot(224)
    # plt.imshow(tracking_image)
    # fig.tight_layout()
    # plt.show()
   
    return tracking_image

# clip.iter_frames is a Python generator that loops through the frames.
# for image in clip.iter_frames():
#     single_image_pipeline( image )

####################
# 5. Process video #
####################

if __name__ == '__main__':
  # Open the input video
  clip = VideoFileClip('project_video.mp4')

  # Process the input video to create the output clip
  output_clip = clip.fl_image(pipeline)

  # Write the output clip
  output_clip.write_videofile('project_output.mp4', audio=False)

