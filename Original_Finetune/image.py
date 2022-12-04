   
import imageio

im = imageio.imread('imageio:laska.png')
print(im.shape)


"""
import imageio
im = imageio.imread('laska.png')
im.shape
(227, 227, 3)    # im is a numpy array
imageio.imwrite('imageio:laska-grey.jpg', im[:, :, 0])
"""


