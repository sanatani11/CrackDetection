import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask, blended_image):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1,3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Crack Mask')
    ax[1].imshow(mask==0)
    # for i in range(classes):
    #     ax[i + 1].set_title(f'Mask (class {i + 1})')
    #     ax[i + 1].imshow(mask == i)
    ax[2].set_title('Blended image')
    ax[2].imshow(blended_image)
    plt.xticks([]), plt.yticks([])
    plt.show()