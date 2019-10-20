import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
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


def extract_edges(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    R = image[:, :, 0]

    binary = np.zeros_like(R)
    combined_binary = np.zeros_like(R)
    binary_gray = np.zeros_like(R)

    binary[(R > 214) & (R <= 255)] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary_gray[(gray > 230) & (gray <= 255)] = 255


    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    binary_hls = np.zeros_like(hls[:, :, 2])
    binary_hls[(hls[:, :, 2] >= 170) & (hls[:, :, 2] <= 245)] = 255
    #binary[(hls[:, :, 1] >= 200)] = 255
    #binary = cv2.Canny(binary, 10,200)
    combined_binary[(binary == 255) | (binary_gray == 255) ] = 255 #| (binary_hls == 255)
    return combined_binary


# cv2.imwrite(os.path.join(save_dir, os.path.basename(fname)), img)

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


    return warped





def moving_avg(x, n):
    mv = np.convolve(x, np.ones(n) / n, mode='valid')
    return np.concatenate(([0 for k in range(n - 1)], mv))

def find_left_line(img):
    # index_half = img.shape[1] // 2
    # img_half = img[:, :index_half]
    # histogram = np.sum(img[:, :index_half], axis=0)
    # histogram = moving_avg(histogram, 150)
    # return histogram.argmax()
    histogram = np.sum(img, axis=0)
    index_half = histogram.size // 2
    histogram = moving_avg(histogram, 30)
    left = histogram[0:index_half].argmax()
    right = histogram[index_half:].argmax() + index_half
    # plt.plot(histogram)
    # plt.show()
    if left == 0:
        return index_half // 2
    else:
        return left

def find_right_line(img):
    # index_half =
    # img_half = img[:, index_half:]
    # histogram = np.sum(img[img.shape[0]:, index_half:], axis=0)
    # histogram = moving_avg(histogram, 150)
    # return histogram.argmax() + index_half
    histogram = np.sum(img, axis=0)
    index_half = histogram.size // 2
    histogram = moving_avg(histogram, 30)
    left = histogram[0:index_half].argmax()
    right = histogram[index_half:].argmax() + index_half
    # plt.plot(histogram)
    # plt.show()
    if right == 0:
        return int(index_half * 1.5)
    else:
        return right

# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits

# View your output

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    line_img = np.zeros_like(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img



def get_lines_from_egdes(edge_image):

    lines = cv2.HoughLinesP(edge_image, 2, np.pi/180, 15, np.array([]), minLineLength=40, maxLineGap=20)

    angle_threshold_min=0.55
    angle_threshold_max =0.8
    leftx=[]
    lefty=[]
    rightx=[]
    righty=[]
    good_lines =[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2 - x1) != 0:
                angle = (y2-y1)/(x2-x1)
                if angle > angle_threshold_min and angle < angle_threshold_max and x1 < edge_image.shape[0]:
                    rightx.append(x1)
                    rightx.append(x2)
                    righty.append(y1)
                    righty.append(y2)
                    good_lines.append(line)
                elif angle < -angle_threshold_min and angle > -angle_threshold_max and x1 > edge_image.shape[0]:
                    leftx.append(x1)
                    leftx.append(x2)
                    lefty.append(y1)
                    lefty.append(y2)
                    good_lines.append(line)

    line_img = draw_lines(edge_image, lines)
    return leftx, lefty, rightx, righty, line_img

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
    minpix = 100

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
    #binary_warped = get_lines_from_egdes(binary_warped)

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
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
    #out_img[lefty, leftx] = [255,0,0]
    #out_img[righty, rightx] =[0,0, 255]
#    leftx, lefty, rightx, righty, out_img = get_lines_from_egdes(out_img)
    return leftx, lefty, rightx, righty, out_img



def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # ym_per_pix = 30 / 720  # meters per pixel in y dimension
    # xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # leftx = leftx * xm_per_pix
    # lefty = lefty * ym_per_pix
    # rightx = rightx * xm_per_pix
    # righty = righty *ym_per_pix
    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx,2)
    right_fit = np.polyfit(righty, rightx,2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    #plt.show()
    #out_img = get_lines_from_egdes(out_img)
    return result, left_fit, right_fit



def process_image(img, mtx, dist):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    image = cv2.GaussianBlur(dst, (5, 5), 0)

    out_img = extract_edges(image)
    warped_img = warp_image(out_img)
    cv2.imshow('raw', warped_img)

    # # leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped_img)
    # cv2.imshow('image_with_threshold_', warped_img)
    # # left_line_approx, right_line_approx = find_initial_lines(warped_img)
    out_img, left_fit, right_fit = fit_polynomial(warped_img)
    #cv2.imshow('out_raw', out_img)
    plt.show()
    # lines = get_lines_from_egdes(warped_img)
    # measure_curvature_pixels(out_img, left_fit, right_fit


    return out_img
import matplotlib.image as mpimg


def test_images():
    ret, mtx, dist, rvecs, tvecs = get_camera_calibration_parameters()

    images = glob.glob(DIR_PATH + '/test_images/*.jpg')
    for fname in images:
        #img = cv2.imread(fname)
        img = mpimg.imread(fname)
        out_img = process_image(img, mtx, dist)
        cv2.waitKey(2500)

        #plt.imshow(out_img)
        #plt.show()
        # out_img = warped_img.copy()
        # # cv2.imshow('img', find_lines_points(warped_img, left_line_approx, right_line_approx, out_img))
        # leftx, lefty, rightx, righty, out_img = find_lines_points(warped_img, left_line_approx, right_line_approx, out_img)
        # result = fit_poly(out_img.shape, leftx, lefty, rightx, righty)

        # cv2.waitKey(20000)

        #
    # cv2.imwrite(os.path.join(save_dir, os.path.basename(fname)), img)



test_images()