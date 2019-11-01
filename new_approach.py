import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class ListOfN:
    def __init__(self, num_of_elements):
        self.data = []
        self.num_of_elements = num_of_elements

    def add(self, val):
        if len(self.data) == self.num_of_elements:
            self.data = self.data[1:] + [val]
        else:
            self.data += [val]

    def get_average(self):
        average_poly = np.array([0.0,0.0,0.0])
        for a, b, c in self.data:
            average_poly += np.array([a, b, c])
        return average_poly/len(self.data)


class Line:
    def __init__(self):
        self.poly = np.array([0, 0, 0])
        self.last_poly = np.array([0, 0, 0])
        self.last_15_polynomials = ListOfN(5)
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension TODO I can se two line segments
        self.xm_per_pix = 3.7 / 600  # meters per pixel in x dimension TODO

    def get_poly_m(self, poly_pix):
        new_polynomial = np.array(poly_pix)
        new_polynomial[0] = poly_pix[0] / (self.ym_per_pix ** 2) * self.xm_per_pix
        new_polynomial[1] = poly_pix[1] / self.ym_per_pix * self.xm_per_pix
        new_polynomial[2] = poly_pix[2] * self.xm_per_pix
        return new_polynomial

    def get_curvature_last_km(self):
        poly_m = self.get_poly_m(self.last_poly)
        y = 30
        return ((1 + (2 * poly_m[0] * y + poly_m[1]) ** 2) ** (3 / 2) / (2 * abs(poly_m[0]))) / 1000

    def get_curvature_km(self):
        poly_m = self.get_poly_m(self.poly)
        y = 30
        return ((1 + (2 * poly_m[0] * y + poly_m[1]) ** 2) ** (3 / 2) / (2 * abs(poly_m[0]))) / 1000

    def get_x_last(self, y):
        return self.last_poly[0] * (y ** 2) + self.last_poly[1] * y + self.last_poly[2]

    def get_x_last_m(self, y):
        return self.get_x_last(y) * self.xm_per_pix

    def get_x(self, y):
        return self.poly[0] * (y ** 2) + self.poly[1] * y + self.poly[2]

    def get_x_m(self, y):
        return self.get_x(y) * self.xm_per_pix

    def add_last_polynomial(self):
        self.last_15_polynomials.add(self.last_poly)
        self.poly = self.last_15_polynomials.get_average()  # TODO average

    def get_line_points(self, ploty):
        try:
            # left_fitx = (xm_per_pix / (ym_per_pix**2)) * left_fit[0] * ploty ** 2 + (xm_per_pix /ym_per_pix) * left_fit[1] * ploty + left_fit[2] * xm_per_pix
            # right_fitx = (xm_per_pix / (ym_per_pix**2)) * right_fit[0] * ploty ** 2 + (xm_per_pix /ym_per_pix) * right_fit[1] * ploty + right_fit[2]* xm_per_pix
            fitx = self.poly[0] * ploty ** 2 + self.poly[1] * ploty + \
                        self.poly[2]

            # left_curverad_Test = calc_curvature(left, y_eval)
            # right_curverad_test = calc_curvature(right, y_eval)
            # print(left_curverad_Test, right_curverad_test )
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            fitx = 1 * ploty ** 2 + 1 * ploty
        return fitx

class RoadLines:
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        self.no_valid_detected = 0
        self.detected = False

    def get_dist_from_center(self):
        return ((self.right_line.get_x(720) + self.left_line.get_x(720))//2 - 1280//2) * 3.7/700

    def get_line_distance(self, pix_position):
        #    np.average((l_fitx + r_fitx)
        x_right_bottom = self.right_line.get_x_m(pix_position)
        x_left_bottom = self.left_line.get_x_m(pix_position)
        dist = x_right_bottom - x_left_bottom
        return dist


    def get_line_distance_last(self, pix_position):
        #    np.average((l_fitx + r_fitx)
        x_right_bottom = self.right_line.get_x_last_m(pix_position)
        x_left_bottom = self.left_line.get_x_last_m(pix_position)
        dist = x_right_bottom - x_left_bottom
        return dist


    def if_valid(self):
        left_line_curve = self.left_line.get_curvature_last_km()
        right_line_curve = self.right_line.get_curvature_last_km()
        line_curve_diff = abs(left_line_curve - right_line_curve)
        #Lines not staright and differenve too big
        if (left_line_curve < 2.5 and right_line_curve < 2.5) and line_curve_diff > 2: #TODO is that true?
            #print("Curvature difference too big")
            return False
        #Curvature too big
        if left_line_curve < 0.2 or right_line_curve < 0.2:
            return False
        dist_bottom = self.get_line_distance_last(720)
        dist_middle = self.get_line_distance_last(350)
        dist_top = self.get_line_distance_last(0)
        if dist_bottom < 3 or dist_middle < 3 or dist_top < 3:
            #print("distance between lines too low")
            return False

        if dist_bottom > 5 or dist_middle > 5 or dist_top > 5:
            #print("distance between lines too big")
            return False

        return True

    # TODO refactor?
    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        if leftx.shape[0] > 0 and lefty.shape[0] > 0 and rightx.shape[0] > 0 and righty.shape[0] > 0:
            self.left_line.last_poly = np.polyfit(lefty, leftx, 2)
            self.right_line.last_poly = np.polyfit(righty, rightx, 2)
            ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
            if leftx.size > 0 and lefty.size > 0:
                try:
                    # left_fitx = (xm_per_pix / (ym_per_pix**2)) * left_fit[0] * ploty ** 2 + (xm_per_pix /ym_per_pix) * left_fit[1] * ploty + left_fit[2] * xm_per_pix
                    # right_fitx = (xm_per_pix / (ym_per_pix**2)) * right_fit[0] * ploty ** 2 + (xm_per_pix /ym_per_pix) * right_fit[1] * ploty + right_fit[2]* xm_per_pix
                    left_fitx = self.left_line.last_poly[0] * ploty ** 2 + self.left_line.last_poly[1] * ploty + \
                                self.left_line.last_poly[
                                    2]
                    right_fitx = self.right_line.last_poly[0] * ploty ** 2 + self.right_line.last_poly[1] * ploty + \
                                 self.right_line.last_poly[2]
                    # left_curverad_Test = calc_curvature(left, y_eval)
                    # right_curverad_test = calc_curvature(right, y_eval)
                    # print(left_curverad_Test, right_curverad_test )
                except TypeError:
                    # Avoids an error if `left` and `right_fit` are still none or incorrect
                    print('The function failed to fit a line!')
                    left_fitx = 1 * ploty ** 2 + 1 * ploty
                    right_fitx = 1 * ploty ** 2 + 1 * ploty

            else:
                return None, None, None
        else:
            return None, None, None
        return left_fitx, right_fitx, ploty

    def fit_poly_binary(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

        self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Colors in the left and right lane regions
        img_zero = np.zeros_like(binary_warped).astype(np.uint8)
        out_img = np.dstack((img_zero, img_zero, img_zero))  # TODO rename
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return out_img

    def search_around_poly(self, binary_warped, left_fit, right_fit):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        # ## Visualization ##
        #out_img = get_line_visualization(binary_warped, left_fitx, right_fitx, ploty)
        img_zero = np.zeros_like(binary_warped).astype(np.uint8)
        out_img = np.dstack((img_zero, img_zero, img_zero))  # TODO rename
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return out_img

    def fit_polynomial(self, binary_warped):
        if not self.detected:
            img = self.fit_poly_binary(binary_warped)
        else:
            img = self.search_around_poly(binary_warped, self.left_line.poly,
                                                                        self.right_line.poly)

        if self.if_valid():
            self.no_valid_detected = 0
            self.detected = True
            self.left_line.add_last_polynomial()
            self.right_line.add_last_polynomial()
        else:
            self.no_valid_detected += 1
            if self.no_valid_detected > 10:
                self.detected = False

        return img


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
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

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
        # pass  # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # except ValueError:
    #    print("Velue error")
    # Avoids an error if the above is not implemented fully
    # pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def extract_edges(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    R = image[:, :, 0]
    G = image[:, :, 1]

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]

    binary_red = np.zeros_like(R)
    binary_saturation = np.zeros_like(R)
    binary_green = np.zeros_like(R)
    binary = np.zeros_like(R)

    binary_red[(R > 200) & (R <= 255)] = 255  # TODO
    binary_green[(G > 200) & (G <= 255)] = 255
    binary_saturation[(S > 200) & (S < 255)] = 255


   # cv2.imshow("saturation", binary_saturation)
    binary[((binary_green == 255) & (binary_red == 255)) | (binary_saturation == 255)] = 255  # () |
    return binary


def warp_image(img):
    height = img.shape[0]
    width = img.shape[1]

    src = np.float32(
        [[(width * 0.4, height * 0.6), (width * 0.05, height), (width * 0.6, height * 0.6), (width * 0.95, height)]])

    src = np.float32(
        [[(width * 0.47, height * 0.55), (width * 0.1, height), (width * 0.53, height * 0.55), (width * 0.9, height)]])

    dst = np.float32([[(0, 0), (0, height), (width, 0), (width, height)]])
    # TODO refactor
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

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped, M



ret, mtx, dist, rvecs, tvecs = get_camera_calibration_parameters()

road_lines = RoadLines()


def get_line_visualization(img, left_fitx, right_fitx, ploty):
    img_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((img_zero, img_zero, img_zero))  # TODO rename
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    return color_warp


def process_image(img):
    global road_lines
    # undist = mpimg.imread(fname)
    # cv2.imshow('raw',undist)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    binary_warped, M = warp_image(img)
    # cv2.imshow("warped", binary_warped)
    # cv2.waitKey()

    image = extract_edges(undist)
    #cv2.imshow('egdes',image)

    binary_warped, M = warp_image(image)
    # cv2.imshow("warped", binary_warped)
    #
    # cv2.waitKey()
    # cv2.waitKey()
    # TODO HERE! the magis is done
    image_with_line_pixels = road_lines.fit_polynomial(binary_warped)
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])

    left_fitx = road_lines.left_line.get_line_points(ploty)
    right_fitx = road_lines.right_line.get_line_points(ploty)
    if left_fitx is not None and right_fitx is not None: #TODO never gona happen
        # left_fit = left_fitx
        # right_fit = right_fitx
        #cv2.imshow("bin", binary_warped)
        #cv2.waitKey()
        # Create an image to draw the lines on

        # image_with_lines = get_line_visualization(binary_warped, left_fitx, right_fitx, ploty)
        result = get_line_visualization(binary_warped, left_fitx, right_fitx, ploty)
        # # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = np.linalg.inv(M)
        newwarp = cv2.warpPerspective(result, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        # plt.imshow(newwarp)
        # plt.show()
        result = cv2.addWeighted(newwarp, 100, undist, 1, 0)
        curve_left = road_lines.left_line.get_curvature_km()
        curve_right = road_lines.right_line.get_curvature_km()
        text_right = "R : %.2f [km]" % curve_right if curve_right < 2.5 else "R : straight"
        text_left = "L : %.2f [km]" % curve_left if curve_left < 2.5 else "L : straight"
        cv2.putText(result, text_right, (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, text_left, (50, 150),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "Dist from center: %.2f [m]:" % road_lines.get_dist_from_center(), (50, 250),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        return result
    else:
        return image


def test_image():
    global road_lines
    images = glob.glob(DIR_PATH + '/test_images/*.jpg')
    for fname in images:
        road_lines = RoadLines()
        image = mpimg.imread(fname)
        result = process_image(image)
        plt.imshow(result)
        plt.show()


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def test_video():
    name = "challenge_video.mp4"
    name = "project_video.mp4"
    white_output = 'test_videos/'+name
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip1 = VideoFileClip(name).subclip(35, 45)
    clip1 = VideoFileClip(name)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


test_video()

#test_image()
