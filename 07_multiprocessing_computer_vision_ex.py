import concurrent.futures
import time
import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def pre_processing(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((50, 50))
    img_dial = cv2.dilate(img_canny, kernel, iterations=100)
    img_threshold = cv2.erode(img_dial, kernel, iterations=100)


images = load_images_from_folder('Resources')

start = time.perf_counter()

count = 0
for image in images:
    count+=1
    pre_processing(image)

finish = time.perf_counter()
print(f'Finished without multiprocessing in {round(finish-start, 2)} second(s)')

start = time.perf_counter()

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(pre_processing, images)

finish = time.perf_counter()
print(f'Finished with multiprocessing in {round(finish-start, 2)} second(s)')



