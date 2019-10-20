import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_camera_calibration_parameters():
    save_dir = os.path.join(DIR_PATH, "output_images/undistorted")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(DIR_PATH + '/camera_cal/*calibration*.jpg')
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def moving_avg(x, n):
    mv = np.convolve(x, np.ones(n) / n, mode='valid')
    return np.concatenate(([0 for k in range(n - 1)], mv))

def find_left_line(img):
    histogram = np.sum(img, axis=0)
    index_half = histogram.size // 2
    histogram = moving_avg(histogram, 30)
    left = histogram[0:index_half].argmax()
    if left == 0:
        return index_half // 2
    else:
        return left

def find_right_line(img):
    histogram = np.sum(img, axis=0)
    index_half = histogram.size // 2
    histogram = moving_avg(histogram, 30)
    right = histogram[index_half:].argmax() + index_half
    if right == 0:
        return int(index_half * 1.5)
    else:
        return right


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        if rightx_current == 0:
            rightx_current = find_right_line(binary_warped[win_y_low:win_y_high][:])
        if leftx_current == 0:
            leftx_current = find_left_line(binary_warped[win_y_low:win_y_high][:])

        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin # Update this
        win_xright_high = rightx_current + margin # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            leftx_current = 0
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            rightx_current = 0


        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        #pass  # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    #try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    #except ValueError:
    #    print("Velue error")
        # Avoids an error if the above is not implemented fully
        #pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # leftx = np.array(leftx *xm_per_pix, dtype='int')
    # lefty = np.array(lefty *ym_per_pix, dtype='int')
    # rightx = np.array(rightx *xm_per_pix,dtype='int')
    # righty = np.array(righty *ym_per_pix,dtype='int')
    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx,2)
    right_fit = np.polyfit(righty, rightx,2)

    # Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    ploty = np.linspace(0, 1299 * ym_per_pix, num=1300 * ym_per_pix)
    ploty = np.linspace(0, 1299 , num=1300 )

    y_eval = np.max(ploty) #TODO move below

    try:
        #left_fitx = (xm_per_pix / (ym_per_pix**2)) * left_fit[0] * ploty ** 2 + (xm_per_pix /ym_per_pix) * left_fit[1] * ploty + left_fit[2] * xm_per_pix
        #right_fitx = (xm_per_pix / (ym_per_pix**2)) * right_fit[0] * ploty ** 2 + (xm_per_pix /ym_per_pix) * right_fit[1] * ploty + right_fit[2]* xm_per_pix
        left_fitx= left_fit[0] * ploty ** 2 +  left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        #left_curverad_Test = calc_curvature(left, y_eval)
        #right_curverad_test = calc_curvature(right, y_eval)
        #print(left_curverad_Test, right_curverad_test )
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    y_eval = np.max(ploty)

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####

    left_curverad = calc_curvature(left_fit, y_eval*ym_per_pix)
    right_curverad = calc_curvature(right_fit, y_eval*ym_per_pix)

    print(left_curverad, right_curverad)

    return left_fitx, right_fitx, ploty

def extract_edges(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    R = image[:, :, 0]

    binary = np.zeros_like(R)
    combined_binary = np.zeros_like(R)
    binary_gray = np.zeros_like(R)

    binary[(R > 230) & (R <= 255)] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary_gray[(gray > 230) & (gray <= 255)] = 255


    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    binary_hls = np.zeros_like(hls[:, :, 2])
    binary_hls[(hls[:, :, 2] >= 170) & (hls[:, :, 2] <= 245)] = 255
    #binary[(hls[:, :, 1] >= 200)] = 255
    #binary = cv2.Canny(binary, 10,200)
    combined_binary[(binary == 255) | (binary_gray == 255) ] = 255 #| (binary_hls == 255)
    return combined_binary

def warp_image(img):
    height = img.shape[0]
    width = img.shape[1]
    offset = 40
    #    "    vertices = np.array([[(width*0.4,height*0.6),( width*0.6,height*0.6), (width* 0.8 , height), (width* 0.2 ,height)]], dtype=np.int32)\n",

    src = np.float32(
        [[(width * 0.4, height * 0.6), (width * 0.1, height), (width * 0.6, height * 0.6), (width * 0.9, height)]])

    dst = np.float32([[(0, 0), (0, height), (width, 0), (width, height)]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)


    return warped, M

def calc_curvature(polynomial, y):

    #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

    return (1+ (2*polynomial[0]*y + polynomial[1])**2)**(3/2)/(2*abs(polynomial[0]))

ret, mtx, dist, rvecs, tvecs = get_camera_calibration_parameters()

def process_image(img):
    # TODO indistort
    #undist = mpimg.imread(fname)
    # cv2.imshow('raw',undist)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    image = extract_edges(undist)
    # cv2.imshow('egdes',image)

    binary_warped, M = warp_image(image)

    # cv2.waitKey()

    left_fitx, right_fitx, ploty = fit_polynomial(binary_warped)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def test_image():
    images = glob.glob(DIR_PATH + '/test_images/*.jpg')
    for fname in images:
        image = mpimg.imread(fname)
        result = process_image(image)
        plt.imshow(result)

        plt.show()
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def test_video():
    white_output = 'project_video.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/project_video.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)