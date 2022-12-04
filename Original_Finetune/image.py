   
import imageio

im = imageio.imread('laska.png')
print(im.shape)


"""
# Input the command in the terminal
$ python image.py

# Show the result
(227, 227, 3)    # im is a numpy array
imageio.imwrite('imageio:laska-grey.jpg', im[:, :, 0])
"""


