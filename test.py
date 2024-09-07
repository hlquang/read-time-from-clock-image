import cv2 as cv
import TimeReader

if __name__ == '__main__':
    tr = TimeReader.TimeReader()
    for i in range(1, 6):
        image_path = f'input/clock{i}.jpg'
        result_image = tr.get_time(image_path, True)
        cv.imwrite(f'output/clock{i}.jpg', result_image)