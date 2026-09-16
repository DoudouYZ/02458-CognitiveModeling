import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from time import sleep

def run_experiment(id:int, intro=False):
    if intro:
        print("Welcome to experiment 3. In a moment the first adapting image will be shown. Please focus your attention on the red dot in the center of the image. After the image changes, you will briefly see the test image. After that image dissappears, please rate it on a scale 1-5")
        sleep(5)
    print("Starting experiment 3")
    test_id = id % 3
    adapting_id = id // 3

    adapting_files = glob.glob(f"exp3_data/adapting_images/adapting_{adapting_id}_*.png") ## TODO: Remember to change this later
    test_files = glob.glob(f"exp3_data/test_images/test_{test_id}_*.png") ## TODO: Remember to change this later

    adapting_files.sort()
    test_files.sort()

    adapting_image = plt.imread(adapting_files[0])
    height, width = adapting_image.shape[:2]
    fig1 = plt.figure()
    plt.imshow(adapting_image)
    plt.plot(width / 2, height / 2, 'ro', markersize=10)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(25)
    plt.close(fig1)

    test_image = plt.imread(test_files[0])
    fig2 = plt.figure()
    plt.imshow(test_image)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close(fig2)

    rating = input("Please rate the image on a scale of 1-5: ")
    return int(rating)
experiments = [0,1,2,3,4,5] 
order = np.random.permutation(experiments)
results = [0]*6
results[0] = run_experiment(order[0], intro=True)
for i in range(1,6):
    results[i] = run_experiment(order[i])
print("Experiment completed. Here are your results:")
for i in range(6):
    print(f"Experiment {order[i]}: Rating {results[i]}")

results_df = pd.DataFrame({'Experiment': order, 'Rating': results})
results_df.to_csv('exp3_results.csv', index=False)