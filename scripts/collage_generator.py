# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:48:26 2022
- Using PIL to concatenate images for a grid thing
@author: Rahul Venugopal
"""
#%% Making two rows of images
import numpy as np
import PIL
from PIL import Image

list_im = ['im1.png', 'im2.png', 'im3.png']
imgs    = [ PIL.Image.open(i) for i in list_im ]

# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
imgs_comb = PIL.Image.fromarray( imgs_comb)
imgs_comb.save( 'Row1.png' )

# for a vertical stacking it is simple: use vstack
imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
imgs_comb = PIL.Image.fromarray( imgs_comb)
imgs_comb.save( 'Col1.png' )

#%% Concatenate two images
# https://note.nkmk.me/en/python-pillow-concat-images/

im7 = Image.open('Row1.png')
im8 = Image.open('Row2.png')

# horizontal concat
dst = Image.new('RGB', (im7.width + im8.width , im7.height))
dst.paste(im7, (0, 0))
dst.paste(im8, (im7.width, 0))

# vertical concat
dst = Image.new('RGB', (im7.width, im8.height + im8.height))
dst.paste(im7, (0, 0))
dst.paste(im8, (0, im7.height))

# save the image
dst.save('FinalColage.png')
