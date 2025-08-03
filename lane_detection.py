import cv2
import numpy as np

# Function to apply perspective transformation
def perspective_transform(image):
    height, width = image.shape[:2]
    src = np.float32([[int(width * 0.45), int(height * 0.65)],
                      [int(width * 0.55), int(height * 0.65)],
                      [int(width * 0.9), height],
                      [int(width * 0.1), height]])
    dst = np.float32([[int(width * 0.1), 0],
                      [int(width * 0.9), 0],
                      [int(width * 0.9), height],
                      [int(width * 0.1), height]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix

# Function to apply color and gradient thresholding
def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# Function to create a histogram to identify lane positions
def histogram_search(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    return left_base, right_base

# Sliding window approach to find lane pixels
def sliding_window_search(binary_warped, left_base, right_base):
    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    left_current = left_base
    right_current = right_base

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        cv2.rectangle(binary_warped, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(binary_warped, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_current = int(np.mean(nonzerox[good_left_inds]))  # Fix applied here
        if len(good_right_inds) > minpix:
            right_current = int(np.mean(nonzerox[good_right_inds]))  # Fix applied here

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


# Fit a polynomial to the lane lines
def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    # Check if any of the arrays are empty
    if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
        print("No lane pixels were detected.")
        return None, None

    # Fit a second degree polynomial to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

# Draw the lane lines onto the original frame
def draw_lanes(original_image, binary_warped, left_fit, right_fit, matrix):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(matrix), (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result

# Main function to process video
def main():
    cap = cv2.VideoCapture('test_video.mp4')  # Change this path to your video file or use '0' for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended or failed to read frame.")
            break

        # Step 1: Perspective Transform
        warped_frame, matrix = perspective_transform(frame)

        # Step 2: Apply Thresholding
        binary_warped = thresholding(warped_frame)

        # Step 3: Histogram Search for Lane Base
        left_base, right_base = histogram_search(binary_warped)

        # Step 4: Sliding Window Search for Lane Pixels
        leftx, lefty, rightx, righty = sliding_window_search(binary_warped, left_base, right_base)

        # Step 5: Fit Polynomial to the Lane Lines
        left_fit, right_fit = fit_polynomial(binary_warped, leftx, lefty, rightx, righty)

        # Step 6: Draw the Lane Lines on the Original Image
        result = draw_lanes(frame, binary_warped, left_fit, right_fit, matrix)

        cv2.imshow("Lane Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
