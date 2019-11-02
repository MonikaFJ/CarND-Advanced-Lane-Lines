## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted]: ./output_images/report_images/undistort_output.jpg "Road Transformed"
[binary]: ./output_images/report_images/binary_example.jpg "Binary Example"
[warped]: ./output_images/report_images/warped.jpg "Warp Example"
[binary_detection]: ./output_images/report_images/binary_sliding_window.jpg 
[example_output]: ./output_images/report_images/example_output.jpg "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! All code can be found in file `advanced_line_find.py`

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This step is implemented in function `get_camera_calibration_parameters()`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction. The assumption is that the camera parameters are known from calibration step.  To undistort the image I'm using `cv2.undistort(img, mtx, dist, None, mtx)` function.
![alt text][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds on RGB and HLS color space to generate a binary image (full implementation in in `advanced_line_finding.py` in function `extract_edges()` ).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

```python
    binary_red[(R > 200) & (R <= 255)] = 255
    binary_green[(G > 200) & (G <= 255)] = 255
    binary_saturation[(S > 200) & (S < 255)] = 255

    binary[((binary_green == 255) & (binary_red == 255)) | (binary_saturation == 255)] = 255
```

![alt text][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`.  The `warp_image()` function takes as inputs an image (`img`) and based on its size calculate the dst and src matrixes.  I chose the hardcode the source and destination points in the following manner:

```python
   src = np.float32(
        [[(width / 2) - 55, height / 2 + 100],
         [((width / 6) - 10), height],
         [(width * 5 / 6) + 60, height],
         [(width / 2 + 55), height / 2 + 100]])
    dst = np.float32(
        [[(width / 4), 0],
         [(width / 4), height],
         [(width * 3 / 4), height],
         [(width * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Input to the pixel identification function is the binary warped image. 

There are two different techniques that I used to detect the line pixels. 

##### No prior knowledge about the position of the line
First one is used when there is no prior knowleadge about the position of the line.  The process consist of following steps:
1. Create histogram from the bottom of the picture and identify the peaks to find where the lines start. I decided to additionally use smoothing to get better results.
1. Split image into 9 segments, in each segments look for line pixels that are within the region of interest originating from the average position of pixels detected in previous segment (in case of the first segment it originates from histogram peaks) (see image below) If there is not enough valid pixels in the segment, the next segment will be detected with histogram. 

![alt text][binary_detection]


The full implementation can be found in function `find_lane_pixels_binary()`.

##### Line was previously detected
Second technique assumes that the approximate position of the line is known, in our case by taking the last detected polynomial. Here the idea is quite simple: all line pixels within +/- 100 pixel in x direction from the lina are marked as line pixels.  The full implementation can be found in function `find_line_pixels_around_poly()`. 

  
Next steps are the same for both algorithms:
 1. Fit second level polynomial to identified line points (np.polyfit(points_y, points_x, 2))
 2. Perform sanity checks to see if the detection is correct
    1. If the lines are between 3 and 5 meters from each other
    2. If the lines has similar curvature
    3. If the lines curvature is not too big
 3. If the lines are valid they are added to the list of detected polynomials. It consist of 5 last properly detected polynomials. For further steps value that is an average from this stored polynomials is used.
 4. If the line is detected incorrectly it increase the `no_valid_detected` counter. If the counter is higher than 5 line is detected with the technique when no prior knowledge is required (histogram)

The full implementation can be found in function `fit_polynomial()`.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Curvature and position is calculated with the polynomial calculated before. Since the polynomial values are calculated in pixels and the values should be in m or km first step is to convert pixels values to SI values.
To do this I'm using formula: 

    x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c
Implemented in function `get_poly_m()`

##### To calculate the curvature I'm using formula:

    curv = ((1 + (2 * a * y + b) ** 2) ** (3 / 2) / (2 * abs(a))) / 10
Implemented in function `get_curvature_km()`

##### To calculate the distance I'm using formula:

    dist = ((position_of_right_line_m + position_of_left_line_m)//2 - img_lenght//2) 
Implemented in function `get_dist_from_center()`
Positions of the lines are calculated with the polynomial.
        
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Overlaying the image with the visualization of the detected road is implemented in function `add_line_visualization()`. The example image can be found below:

![alt text][example_output]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For sure one thing that can be improved is the image transformation. With current implementation the lines that are too bended might be outside the region of interest. The same problem might occur when the car is not in the center between lines.

Second improvement could be done in line detection. Right now I'm only using Red and Green channel from RGB image and Saturation chanel from HLS image. Potentially other color representations and image transformations (Canny detection, Sobel and other techniques) might improve the outcome. 

There are also several code optimization that can be applied (e.g. calculate src and dst matrices only once), but they should not significantly affect the program execution time.