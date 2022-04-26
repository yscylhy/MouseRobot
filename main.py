import mouse
import os
import time
import PIL.ImageGrab
import numpy as np
import imageio
import cv2 as cv
import matplotlib.pylab as plt


def safe_click():
    pos = mouse.get_position()
    time.sleep(1)
    pos_check = mouse.get_position()
    if pos == pos_check:
        mouse.click('left')
    else:
        exit(-1)


def get_grad_feature(image):
    dx, dy = np.gradient(image)
    return (dx**2 + dy**2)**0.5


def detect_object_cv2(screen, target):
    screenshot = (np.mean(screen, axis=2)).astype(np.float32)
    target = (np.mean(target, axis=2)).astype(np.float32)

    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    method = eval(methods[5])

    res = cv.matchTemplate(screenshot, target, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    top_left = [top_left[1], top_left[0]]
    click_pos = [x+y//2 for x, y in zip(top_left, target.shape)]

    return click_pos


def detect_object(screen, target):
    screenshot = get_grad_feature(np.mean(screen, axis=2))
    target = get_grad_feature(np.mean(target, axis=2))
    target -= np.mean(target)
    target_canvas = np.zeros_like(screenshot)
    target_canvas[:target.shape[0], :target.shape[1]] = target
    screenshot -= np.mean(screenshot)
    cross_map = np.real(np.fft.ifft2(np.fft.fft2(screenshot)*np.conjugate(np.fft.fft2(target_canvas))))
    pos = np.unravel_index(np.argmax(cross_map), screenshot.shape)
    click_pos = [x+y//2 for x, y in zip(pos, target.shape)]

    return click_pos


def calibrate():
    print('Start Calibration...')
    max_pos = 10000
    screenshot = PIL.ImageGrab.grab()
    mouse.move(max_pos, max_pos, absolute=True, duration=2)
    pos_x, pos_y = mouse.get_position()
    ratio_x, ratio_y = screenshot.width/pos_x, screenshot.height/pos_y
    assert ratio_x == ratio_y, "Strange screen..."
    print('Calibration succeed.')
    return ratio_x


if __name__ == "__main__":
    cycle_number = 10
    landmark_folder = 'landmarks'
    image_list = os.listdir(landmark_folder)
    image_list.sort(key=lambda x: int(x.split('.')[0]))

    ratio = calibrate()

    print("Open Chrome Browser. \nPress Enter to continue...")

    for cycle_idx in range(cycle_number):
        print(f'start {cycle_idx}th cycle')
        tic = time.time()

        for step_idx in range(len(image_list)):
            print(f'Step {step_idx}')
            screenshot = PIL.ImageGrab.grab()
            target = imageio.imread(os.path.join(landmark_folder, f'{step_idx+1}.PNG'))
            click_pos_y, click_pos_x = detect_object(screenshot, target)
            # click_pos_y, click_pos_x = detect_object_cv2(screenshot, target)
            mouse.move(click_pos_x/ratio, click_pos_y/ratio, absolute=True, duration=1)
            safe_click()

