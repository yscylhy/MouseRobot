import mouse
import os
import time
import PIL.ImageGrab
import numpy as np
import imageio


def safe_click():
    pos = mouse.get_position()
    time.sleep(1)
    pos_check = mouse.get_position()
    if pos == pos_check:
        mouse.click('left')
        time.sleep(2)
    else:
        exit(-1)


def move_mouse(target_pos):
    speed = 1000
    cur_pos = mouse.get_position()
    move_time = sum((x1 - x2)**2 for x1, x2 in zip(cur_pos, target_pos))**0.5/speed
    mouse.move(target_pos[0], target_pos[1], absolute=True, duration=move_time)
    return


def get_grad_feature(image):
    dx, dy = np.gradient(image)
    grad = (dx**2 + dy**2)**0.5
    return grad


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
            move_mouse([click_pos_x/ratio, click_pos_y/ratio])
            safe_click()

