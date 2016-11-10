from PIL import Image

import scipy
import numpy as np
import os
import sys
from sklearn.decomposition import PCA

filename = os.path.expanduser(sys.argv[1]) 

image = scipy.misc.imread(filename)[:,:, :3]

original_shape = image.shape

pixels = image.reshape((original_shape[0]*original_shape[1], 3))

pca = PCA(n_components = 1)

first_pca_component = np.squeeze(pca.fit_transform(pixels))

pixels_by_first_component = pixels[np.argsort(first_pca_component)]

print pixels_by_first_component.shape

n_rows = original_shape[0]
n_columns = original_shape[1]

new_image = np.zeros_like(image)

old_column = None

for column in range(n_columns):
  if column % 10 == 0:
    print 'processing column %s of %s' % (column, n_columns)
  pixels_in_column = pixels_by_first_component[n_rows*column:(column+1)*n_rows]
  first_column_component = np.squeeze(pca.fit_transform(pixels_in_column))
  new_column = pixels_in_column[np.argsort(first_column_component)]
  
  # Now figure out which direction will match the previous column better
  if old_column is not None:
    distance = np.mean(np.sqrt(np.sum((new_column - old_column)**2, 1)))
    flipped_distance = np.mean(np.sqrt(np.sum((new_column[-1::-1] - old_column)**2, 1)))
    if flipped_distance < distance:
      new_column = new_column[-1::-1]   
  old_column = new_column

  new_image[:, column, :] = new_column


im = Image.fromarray(new_image)
im.save("output.png")
  
  
  
