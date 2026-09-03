import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

images = glob.glob('FinalData/*.jpg')
images = images[:10]  # Limit to first 10 images for testing
show_counts = np.zeros(len(images))
ratings = np.zeros((len(images), 2))  # 2 columns: rating on 1st showing, rating on 2nd showing
name = "s224202" # REMEMBER: update this to your own student ID
plt.ion()
fig, ax = plt.subplots()

while any(show_counts < 2):
    image = np.random.choice(range(len(images)))
    if show_counts[image] < 2:
        image_view = images[image]
        ax.clear()
        ax.imshow(plt.imread(image_view))
        ax.axis('off')
        fig.canvas.draw()
        plt.pause(0.001)  # let the window redraw without blocking

        try:
            rating = int(input("Please rate the image (1-5): "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
            continue
        if rating < 1 or rating > 5:
            print("Invalid rating. Please enter a number between 1 and 5.")
            continue
        if show_counts[image] == 0:
            ratings[image, 0] = rating
        else:
            ratings[image, 1] = rating

        show_counts[image] += 1

plt.close(fig)

df = pd.DataFrame({
    'FileName': images,
    'Rating1': ratings[:, 0],
    'Rating2': ratings[:, 1],
})
df.to_csv(f'{name}_ratings.csv', index=False)
print(df.head())