import cv2 as cv
import numpy as np
import math


class TimeReader:
    def __init__(self):
        self.image = None       # Store image after resizing
        self.thresholded = None # Store image after applying thresholding to resized image

    # Resize image without distortion
    # Input: image - original image
    # Output: resized image
    def resize(self, image):
        height, width, _ = image.shape
        scale_factor = 1000 / max(height, width)
        image = cv.resize(
            image, (int(width * scale_factor), int(height * scale_factor))
        )
        return image

    # Apply thresholding technique to resized image
    # Input: image - resized image
    # Output: thresholded image
    def apply_thresholding(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)[1]
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv.dilate(thresh, kernel, iterations=1)
        return thresh

    # Detect clock face using Hough Circle Transform, get center coordinates
    # and radius of detected clock face (or circle)
    # Output: x coordinate, y coordinate and radius
    def get_clock_face_info(self):
        image = self.thresholded
        clock_face = cv.HoughCircles(
            image, cv.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=50
        )[0, 0]
        clock_face = np.uint16(np.around(clock_face))
        center_x, center_y, radius = clock_face[0], clock_face[1], clock_face[2]
        return center_x, center_y, radius

    # Get angle between the input line and positive x-axis
    # Input: line - 2 endpoints coordinates of the line
    # Output: angle between the line and positive x-axis (in radian)
    def get_angle_horizontal(self, line):
        x1, y1, x2, y2 = line
        return math.atan2(y1 - y2, x1 - x2)

    # Get length of the input line
    # Input: line - 2 endpoints coordinates of the line
    # Output: length of the line
    def get_length(self, line):
        x1, y1, x2, y2 = line
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Get distance between 2 input lines
    # Input: line1 - 2 endpoints coordinates of the first line
    #        line2 - 2 endpoints coordinates of the second line
    # Output: distance between 2 lines
    def get_distance(self, line1, line2):
        line1_x1, line1_y1, line1_x2, line1_y2 = line1
        line2_x1, line2_y1, _, _ = line2

        vector1 = np.array([line1_x2 - line1_x1, line1_y2 - line1_y1])
        vector = np.array([line2_x1 - line1_x1, line2_y1 - line1_y1])

        distance = abs(np.cross(vector1, vector)) / np.linalg.norm(vector1)

        return distance

    # Get thickness of line based on the distance between 2 lines that form
    # the largest angle (in radian)
    # Input: lines - list of lines (that represent the same clock hand)
    # Output: thickness of the line
    def get_thickness(self, lines):
        max_angle_diff = 0
        max_angle_lines = None
        for i in range(len(lines) - 1):
            for j in range(i + 1, len(lines)):
                angle1 = self.get_angle_horizontal(lines[i])
                angle2 = self.get_angle_horizontal(lines[j])
                angle_diff = abs(angle1 - angle2)

                if angle_diff > max_angle_diff:
                    max_angle_diff = angle_diff
                    max_angle_lines = (lines[i], lines[j])
        if max_angle_lines is not None:
            return self.get_distance(max_angle_lines[0], max_angle_lines[1])
        else:
            return 0

    # Identify 3 clock hands: hour hand, minute hand, second hand;
    # and return the line that represents each respective hand.
    # Output: 3 lines/2 endpoints coordinates that represents 3 clock hands
    def get_clock_hands(self):
        image = self.thresholded
        contours = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
        x, y, r = self.get_clock_face_info()
        for contour in contours:
            if cv.pointPolygonTest(contour, (x, y), True) >= 0:
                arms = np.zeros_like(image)
                cv.drawContours(arms, [contour], 0, (1), -1)
                arms = (255 * arms).clip(0, 255).astype(np.uint8)
                arms_thin = cv.ximgproc.thinning(arms)
                lines = cv.HoughLinesP(arms_thin, 1, np.pi / 180, 70, None, 100, r)
                if lines is not None:
                    clock_hands = [line for [line] in lines]
                    i = 0
                    while i < len(clock_hands):
                        line1 = clock_hands[i]
                        angle1 = self.get_angle_horizontal(line1)
                        length1 = math.sqrt(self.get_length(line1))
                        j = i + 1
                        while j < len(clock_hands):
                            line2 = clock_hands[j]
                            angle2 = self.get_angle_horizontal(line2)
                            length2 = math.sqrt(self.get_length(line2))

                            if abs(angle1 - angle2) < 0.01:
                                if length1 > length2:
                                    del clock_hands[j]
                                else:
                                    del clock_hands[i]
                                    i -= 1
                                    break
                            else:
                                j += 1
                        i += 1
                    edges = cv.Canny(arms, 10, 10)
                    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 25, None, 50, 200)
                    clock_hands_group = [
                        [clock_hands[0]],
                        [clock_hands[1]],
                        [clock_hands[2]],
                    ]
                    for [line] in lines:
                        for i in range(3):
                            if (
                                abs(
                                    self.get_angle_horizontal(line)
                                    - self.get_angle_horizontal(clock_hands[i])
                                )
                                < 0.1
                                and abs(
                                    self.get_length(line)
                                    - self.get_length(clock_hands[i])
                                )
                                < 200
                            ):
                                clock_hands_group[i].append(line)
                                break
                    clock_hands_group.sort(key=self.get_thickness)
                    second_hand = clock_hands_group[0][0]
                    hour_hand = None
                    minute_hand = None
                    if self.get_length(clock_hands_group[1][0]) < self.get_length(
                        clock_hands_group[2][0]
                    ):
                        hour_hand = clock_hands_group[1][0]
                        minute_hand = clock_hands_group[2][0]
                    else:
                        hour_hand = clock_hands_group[2][0]
                        minute_hand = clock_hands_group[1][0]

                    return hour_hand, minute_hand, second_hand

    # Get coordinates to draw a rectange that surrounds the input line
    # Input: line - 2 endpoints coordinates of the line
    # Output: coordinates of the rectangle
    def get_rectangle_coordinate(self, line):
        x1, y1, x2, y2 = line
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return x, y, w, h

    # Draw rectangles surrounding 3 clock hands
    # Input: image - resized image
    def mark_clock_hands(self, image):
        hour_hand, minute_hand, second_hand = self.get_clock_hands()
        x, y, w, h = self.get_rectangle_coordinate(hour_hand)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv.putText(
            image,
            "hour",
            (hour_hand[0], hour_hand[1]),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        x, y, w, h = self.get_rectangle_coordinate(minute_hand)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv.putText(
            image,
            "minute",
            (minute_hand[0], minute_hand[1]),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        x, y, w, h = self.get_rectangle_coordinate(second_hand)
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv.putText(
            image,
            "second",
            (second_hand[0], second_hand[1]),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    # Get directed vector of the input line
    # Input: line - 2 endpoints coordinates of the line
    # Output: list that represents the directed vector
    def get_directed_vector(self, line):
        x1, y1, x2, y2 = line
        center_x, center_y, _ = self.get_clock_face_info()
        length1 = np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2)
        length2 = np.sqrt((x2 - center_x) ** 2 + (y2 - center_y) ** 2)
        if max(length1, length2) == length1:
            return [x1 - center_x, y1 - center_y]
        else:
            return [x2 - center_x, y2 - center_y]

    # Calculate dot product
    def dot(self, vector1, vector2):
        return vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate determinant
    def det(self, vector1, vector2):
        return vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # Get angle between the input line and the vertical line
    # to compute the time
    # Input: line - 2 endpoints coordinates of the line
    # Output: angle between the line and vertical line (in degree)
    def get_angle_vertical(self, line):
        u = [0, -100]
        v = self.get_directed_vector(line)
        angle = math.atan2(self.det(u, v), self.dot(u, v))
        angle_degree = math.degrees(angle)
        if angle < 0:
            return angle_degree + 360
        else:
            return angle_degree

    # This is the method that contains (almost) all the computation
    # Read image from the path, process it and return the time
    # Input: image_path: path to the image to retrieve the time
    #        return_image: if True, display the time on the image and draw rectangles around 3 clock hands
    #                      else, return the time as a string
    # Output: time in string type or image
    def get_time(self, image_path, return_image=False):
        self.image = cv.imread(image_path)
        self.image = self.resize(self.image)
        self.thresholded = self.apply_thresholding(self.image)
        hour_hand, minute_hand, second_hand = self.get_clock_hands()
        hour_angle = self.get_angle_vertical(hour_hand)
        minute_angle = self.get_angle_vertical(minute_hand)
        second_angle = self.get_angle_vertical(second_hand)
        hour = hour_angle / 30
        minute = minute_angle / 6
        second = second_angle / 6
        if round(hour) * 30 - hour_angle <= 8 and (
            352 < minute_angle < 360 or minute_angle < 92
        ):
            hour = round(hour)
        if hour_angle - hour * 30 <= 8 and 352 < minute_angle < 360:
            minute = 0
        if round(minute) * 6 - minute_angle <= 8 and second_angle < 8:
            minute = round(minute)
            if minute == 60:
                minute = 0
        if minute_angle - minute * 30 <= 8 and 352 < second_angle < 360:
            second = 0
        hour = int(hour)
        minute = int(minute)
        second = int(second)
        time = f"{hour:02d}:{minute:02d}:{second:02d}"
        if return_image == False:
            return time
        else:
            self.mark_clock_hands(self.image)
            cv.putText(
                self.image, time, (10, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10
            )
            cv.putText(
                self.image,
                time,
                (10, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 255),
                2,
            )
            return self.image


# tr = TimeReader()
# tr.get_time("images/clock11.jpg", True)
