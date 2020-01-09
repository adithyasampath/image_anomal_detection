from timeit import default_timer as timer
import numpy as np 
import cv2

def decode_image_opencv(image,max_height=800,swapRB=True,imagenet_mean = (0,0,0)):
  ### Going to create image vector via OpenCV
  #todo https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html
  start = timer()
  #image = cv2.imread(img_path,1)
  #print("original image shape=",image.shape)
  (h, w) = image.shape[:2]
  #print("Scale factor=", h/max_height) #we are currenly drawing in original so this is not relevant
  image = image_resize(image,height=max_height)
  org  = image
  #IMAGENET_MEAN = (103.939, 116.779, 123.68)
  # for certain architectues this is subtracted
  # please check/test your NW 
  # more details https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
  #IMAGENET_MEAN = (103.939, 116.779, 123.68)
  image = cv2.dnn.blobFromImage(image, scalefactor=1.0,mean=imagenet_mean, swapRB=swapRB)
  # this gives   shape as  (1, 3, 480, 640))
  image = np.transpose(image, (0, 2, 3, 1))
  # we get it after transpose as ('Input shape=', (1, 480, 640, 3))
  #print("resized image shape=",image.shape)
  # for original image we take the first image, (the first dim is number of images)
  #org = image[0,:,:,:] 
  #print("Draw shape=",org.shape)
  end = timer()
  #print("decode time=",end - start)
  return image,org

#https://stackoverflow.com/a/44659589/429476
# It is important to resize without loosing the aspect ratio for 
# good detection
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
  

def create_dummy_image(width=1067,height=800,channels=3):
    #Create a random numpy array
    return np.random.rand(height,width, channels).astype('f')
