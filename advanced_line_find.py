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
        self.last_5_polynomials = ListOfN(5)
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
        self.last_5_polynomials.add(self.last_poly)
        self.poly = self.last_5_polynomials.get_average()

    def get_line_points(self, ploty):
        try:
            line_x = self.poly[0] * ploty ** 2 + self.poly[1] * ploty + \
                        self.poly[2]
        except TypeError:
            print('The function failed to fit a line!')
            line_x = 1 * ploty ** 2 + 1 * ploty
        return line_x


class RoadLines:
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        self.no_valid_detected = 0
        self.detected = False

    def get_dist_from_center(self):
        return ((self.right_line.get_x(720) + self.left_line.get_x(720))//2 - 1280//2) * 3.7/700

    def get_line_distance_last(self, pix_position):
        x_right_bottom = self.right_line.get_x_last_m(pix_position)
        x_left_bottom = self.left_line.get_x_last_m(pix_position)
        return x_right_bottom - x_left_bottom

    def is_detection_valid(self):
        left_line_curve = self.left_line.get_curvature_last_km()
        right_line_curve = self.right_line.get_curvature_last_km()
        line_curve_diff = abs(left_line_curve - right_line_curve)
        #Lines not staright and differenve too big
        if (left_line_curve < 2.5 and right_line_curve < 2.5) and line_curve_diff > 2:
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

    def fit_poly(self, leftx, lefty, rightx, righty):
        if leftx.shape[0] > 0 and lefty.shape[0] > 0 and rightx.shape[0] > 0 and righty.shape[0] > 0:
            self.left_line.last_poly = np.polyfit(lefty, leftx, 2)
            self.right_line.last_poly = np.polyfit(righty, rightx, 2)


    def find_line_pixels_around_poly(self, binary_warped, left_fit, right_fit):
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
        return leftx, lefty, rightx, righty
        self.fit_poly(leftx, lefty, rightx, righty)

        # ## Visualization ##
        img_zero = np.zeros_like(binary_warped).astype(np.uint8)
        out_img = np.dstack((img_zero, img_zero, img_zero))
        out_img[left, nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return out_img

    def fit_polynomial(self, binary_warped):
        if not self.detected:
            leftx, lefty, rightx, righty = find_lane_pixels_binary(binary_warped)
        else:
            leftx, lefty, rightx, righty = self.find_line_pixels_around_poly(binary_warped, self.left_line.poly,
                                                                             self.right_line.poly)
        self.fit_poly(leftx, lefty, rightx, righty)
        ## Visualization ##
        img_zero = np.zeros_like(binary_warped).astype(np.uint8)
        out_img = np.dstack((img_zero, img_zero, img_zero))
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        if self.is_detection_valid():
            self.no_valid_detected = 0
            self.detected = True
            self.left_line.add_last_polynomial()
            self.right_line.add_last_polynomial()
        else:
            self.no_valid_detected += 1
            if self.no_valid_detected > 5:
                self.detected = False

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = road_lines.left_line.get_line_points(ploty)
        right_fitx = road_lines.right_line.get_line_points(ploty)
        add_line_visualization(out_img, left_fitx, right_fitx, ploty)

        return left_fitx, right_fitx, out_img


def get_camera_calibration_parameters():
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

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


def find_lane_pixels_binary(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
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


    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def extract_edges(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    R = image[:, :, 0]
    G = image[:, :, 1]

    binary_red = np.zeros_like(R)
    binary_saturation = np.zeros_like(R)
    binary_green = np.zeros_like(R)
    binary = np.zeros_like(R)

    binary_red[(R > 200) & (R <= 255)] = 255
    binary_green[(G > 200) & (G <= 255)] = 255
    binary_saturation[(S > 200) & (S < 255)] = 255

    binary[((binary_green == 255) & (binary_red == 255)) | (binary_saturation == 255)] = 255
    return binary


def warp_image(img):
    height = img.shape[0]
    width = img.shape[1]

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


def add_line_visualization(img, left_line_x, right_line_x, ploty):
    pts_left = np.array([np.transpose(np.vstack([left_line_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(img, np.int_([pts]), (0, 255, 0))

def add_text(img):
    curve_left = road_lines.left_line.get_curvature_km()
    curve_right = road_lines.right_line.get_curvature_km()
    text_right = "R : %.2f [km]" % curve_right if curve_right < 2.5 else "R : straight"
    text_left = "L : %.2f [km]" % curve_left if curve_left < 2.5 else "L : straight"
    cv2.putText(img, text_right, (50, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, text_left, (50, 150),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Dist from center: %.2f [m]:" % road_lines.get_dist_from_center(), (50, 250),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)


def get_processed_image(img):
    global road_lines
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    image = extract_edges(undist)
    binary_warped, M = warp_image(image)
    left_fitx, right_fitx, image_with_lines = road_lines.fit_polynomial(binary_warped)


    if left_fitx is not None and right_fitx is not None: #TODO never gona happen
        Minv = np.linalg.inv(M)
        unwarp = cv2.warpPerspective(image_with_lines, Minv, (image.shape[1], image.shape[0]))
        result = cv2.addWeighted(unwarp, 100, undist, 1, 0)
        add_text(result)
        return result
    else:
        return image


def test_image():
    global road_lines
    images = glob.glob(DIR_PATH + '/test_images/*.jpg')
    for fname in images:
        road_lines = RoadLines()
        image = mpimg.imread(fname)
        result = get_processed_image(image)
        plt.imshow(result)
        plt.show()


# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML


def test_video():
    name = "challenge_video.mp4"
    name = "project_video.mp4"
    white_output = 'test_videos/'+name
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
#    clip1 = VideoFileClip(name)
#    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
#    white_clip.write_videofile(white_output, audio=False)

ret, mtx, dist, rvecs, tvecs = get_camera_calibration_parameters()

road_lines = RoadLines()

#test_video()
test_image()
