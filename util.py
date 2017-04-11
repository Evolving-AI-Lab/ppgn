import numpy as np
import scipy.misc
import subprocess

def normalize(img, out_range=(0.,1.), in_range=None):
    if not in_range:
        min_val = np.min(img)
        max_val = np.max(img)
    else:
        min_val = in_range[0]
        max_val = in_range[1]

    result = np.copy(img)
    result[result > max_val] = max_val
    result[result < min_val] = min_val
    result = (result - min_val) / (max_val - min_val) * (out_range[1] - out_range[0]) + out_range[0]
    return result

def deprocess(images, out_range=(0.,1.), in_range=None):
    num = images.shape[0]
    c = images.shape[1]
    ih = images.shape[2]
    iw = images.shape[3]

    result = np.zeros((ih, iw, 3))

    # Normalize before saving
    result[:] = images[0].copy().transpose((1,2,0))
    result = normalize(result, out_range, in_range)
    return result

def get_image_size(data_shape):
    '''
    Return (227, 227) from (1, 3, 227, 227) tensor.
    '''
    if len(data_shape) == 4:
        return data_shape[2:] 
    else:
        raise Exception("Data shape invalid.")

def save_image(img, name):
    '''
    Normalize and save the image.
    '''
    img = img[:,::-1, :, :] # Convert from BGR to RGB
    output_img = deprocess(img, in_range=(-120,120))                
    scipy.misc.imsave(name, output_img)

def write_label_to_img(filename, label):
    # Add a label below each image via ImageMagick
    subprocess.call(["convert %s -gravity south -splice 0x10 %s" % (filename, filename)], shell=True)
    subprocess.call(["convert %s -append -gravity Center -pointsize %s label:\"%s\" -border 0x0 -append %s" %
         (filename, 30, label, filename)], shell=True)


def convert_words_into_numbers(vocab_file, words):
    # Load vocabularty
    f = open(vocab_file, 'r')
    lines = f.read().splitlines()

    numbers = [ lines.index(w) + 1 for w in words ]
    numbers.append( 0 )     # <unk> 
    return numbers

def get_mask():
    '''
    Compute the binary mask to be used for inpainting experiments.
    '''

    image_shape = (3, 227, 227)

    # Make a blob of noise in the center
    mask = np.zeros(image_shape) 
    mask_neg = np.ones(image_shape) 

    # width and height of the mask
    w, h = (100, 100)

    # starting and ending positions of mask
    max_x, max_y = image_shape[1] - w, image_shape[2] - h
    x0 = np.random.randint(low=0, high=max_x)
    x1 = np.min([ image_shape[1], x0 + w ])

    y0 = np.random.randint(low=0, high=max_y)
    y1 = np.min([ image_shape[2], y0 + h ])

    for y in np.arange(x0, x1):
      for x in np.arange(y0, y1):
          mask [ :, x, y ] = 1
          mask_neg [ :, x, y ] = 0

    return mask, mask_neg


def compute_topleft(input_size, output_size):
    '''
    Compute the offsets (top, left) to crop the output image if its size does not match that of the input image.
    The output size is fixed at 256 x 256 as the generator network is trained on 256 x 256 images.
    However, the input size often changes depending on the network.
    '''

    assert len(input_size) == 2, "input_size must be (h, w)"
    assert len(output_size) == 2, "output_size must be (h, w)"

    topleft = ((output_size[0] - input_size[0])/2, (output_size[1] - input_size[1])/2)
    return topleft


def apply_mask(img, mask, context):

    assert len(img.shape) == 4
    assert img.shape[0] == 1
    assert img.shape[1] == 3

    # Mask out a patch (defined by the binary "mask")
    img[0] *= mask
    img += context

    return img


def stitch(left, right):
    '''
    Stitch two images together horizontally.
    '''
    
    assert len(left.shape) == 4
    assert len(right.shape) == 4
    assert left.shape[0] == 1
    assert right.shape[0] == 1

    # Save final image and the masked image
    image_size = right.shape[2]
    separator_width = 1
    canvas_size = image_size * 2 + separator_width
    output = np.zeros( (1, 3, image_size, canvas_size) )
    output.fill(255.0) 
    output[:,:,:image_size,:image_size] = left
    output[:,:,:,image_size + separator_width:] = right

    return output
