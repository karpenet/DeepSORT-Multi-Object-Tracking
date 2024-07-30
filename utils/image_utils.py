import matplotlib.pyplot as plt

def visualize_images(input_images):
    fig = plt.figure(figsize=(150, 75))
    for i in range(len(input_images)):
        fig.add_subplot(1, len(input_images), i + 1)
        plt.imshow(input_images[i])
    plt.show()
