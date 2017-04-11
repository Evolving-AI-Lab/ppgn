#!/usr/bin/env python
'''
Anh Nguyen <anh.ng8@gmail.com>
2016
'''

import os, sys
os.environ['GLOG_minloglevel'] = '2'    # suprress Caffe verbose prints

import settings
sys.path.insert(0, settings.caffe_root)
import caffe

import numpy as np
from numpy.linalg import norm
import scipy.misc, scipy.io
import argparse 
import util
from sampler import Sampler

if settings.gpu:
    caffe.set_mode_gpu() # sampling on GPU 

class ClassConditionalSampler(Sampler):

    def __init__ (self):
        # Load the list of class names
        with open(settings.synset_file, 'r') as synset_file:
            self.class_names = [ line.split(",")[0].split(" ", 1)[1].rstrip('\n') for line in synset_file.readlines()]

        # Hard-coded list of layers that has been tested 
        self.fc_layers = ["fc6", "fc7", "fc8", "loss3/classifier", "fc1000", "prob"]
        self.conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]


    def forward_backward_from_x_to_condition(self, net, end, image, condition):
        '''
        Forward and backward passes through 'net', the condition model p(y|x), here an image classifier. 
        '''

        unit = condition['unit']
        xy = condition['xy']

        dst = net.blobs[end]

        acts = net.forward(data=image, end=end)
        one_hot = np.zeros_like(dst.data)

        # Get the activations
        if end in self.fc_layers:
            layer_acts = acts[end][0]
        elif end in self.conv_layers:
            layer_acts = acts[end][0, :, xy, xy]

        best_unit = layer_acts.argmax()     # highest probability unit

        # Compute the softmax probs by hand because it's handy in case we want to condition on hidden units as well
        exp_acts = np.exp(layer_acts - np.max(layer_acts))
        probs = exp_acts / (1e-10 + np.sum(exp_acts, keepdims=True))

        # The gradient of log of softmax, log(p(y|x)), reduces to:
        softmax_grad = 1 - probs.copy()

        obj_prob = probs.flat[unit]

        # Assign the gradient 
        if end in self.fc_layers:
            one_hot.flat[unit] = softmax_grad[unit]
        elif end in self.conv_layers:
            one_hot[:, unit, xy, xy] = softmax_grad[unit]
        else:
            raise Exception("Invalid layer type!")
        
        dst.diff[:] = one_hot

        # Backpropagate the gradient to the image layer 
        diffs = net.backward(start=end, diffs=['data'])
        g = diffs['data'].copy()

        dst.diff.fill(0.)   # reset objective after each step

        # Info to be printed out in the below 'print_progress' method
        info = {
            'best_unit': best_unit,
            'best_unit_prob': probs.flat[best_unit]
        }
        return g, obj_prob, info 


    def get_label(self, condition):
        unit = condition['unit']
        return self.class_names[unit]


    def print_progress(self, i, info, condition, prob, grad):
        print "step: %04d\t max: %4s [%.2f]\t obj: %4s [%.2f]\t norm: [%.2f]" % ( i, info['best_unit'], info['best_unit_prob'], condition['unit'], prob, norm(grad) )

def get_code(encoder, path, layer, mask=None):
    '''
    Push the given image through an encoder (here, AlexNet) to get a code.
    '''

    # set up the inputs for the net: 
    image_size = encoder.blobs['data'].shape[2:]    # (1, 3, 227, 227)
    images = np.zeros_like(encoder.blobs["data"].data, dtype='float32')

    in_image = scipy.misc.imread(path)
    in_image = scipy.misc.imresize(in_image, (image_size[0], image_size[1]))
    images[0] = np.transpose(in_image, (2, 0, 1))   # convert to (3, 227, 227) format

    data = images[:,::-1]   # convert from RGB to BGR

    # subtract the ImageNet mean
    image_mean = scipy.io.loadmat('misc/ilsvrc_2012_mean.mat')['image_mean'] # (256, 256, 3)
    topleft = util.compute_topleft(image_size, image_mean.shape[:2])
    image_mean = image_mean[topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]   # crop the image mean
    data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0)    # mean is already BGR

    if mask is not None:
        data *= mask

    # initialize the encoder
    encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)

    # extract the features
    encoder.forward(data=data)
    features = encoder.blobs[layer].data.copy()

    return features, data

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--units', metavar='units', type=str, help='an unit to visualize e.g. [0, 999]')
    parser.add_argument('--n_iters', metavar='iter', type=int, default=10, help='Number of sampling steps per each unit')
    parser.add_argument('--threshold', metavar='w', type=float, default=-1.0, nargs='?', help='The probability threshold to decide whether to keep an image')
    parser.add_argument('--save_every', metavar='save_iter', type=int, default=1, help='Save a sample every N iterations. 0 to disable saving')
    parser.add_argument('--reset_every', metavar='reset_iter', type=int, default=0, help='Reset the code every N iterations')
    parser.add_argument('--lr', metavar='lr', type=float, default=2.0, nargs='?', help='Learning rate')
    parser.add_argument('--lr_end', metavar='lr', type=float, default=-1.0, nargs='?', help='Ending Learning rate')
    parser.add_argument('--epsilon1', metavar='lr', type=float, default=1.0, nargs='?', help='Prior')
    parser.add_argument('--epsilon2', metavar='lr', type=float, default=1.0, nargs='?', help='Condition')
    parser.add_argument('--epsilon3', metavar='lr', type=float, default=1.0, nargs='?', help='Noise')
    parser.add_argument('--epsilon4', metavar='lr', type=float, default=0.0, nargs='?', help='Context')
    parser.add_argument('--seed', metavar='n', type=int, default=0, nargs='?', help='Random seed')
    parser.add_argument('--xy', metavar='n', type=int, default=0, nargs='?', help='Spatial position for conv units')
    parser.add_argument('--opt_layer', metavar='s', type=str, help='Layer at which we optimize a code')
    parser.add_argument('--act_layer', metavar='s', type=str, default="fc8", help='Layer at which we activate a neuron')
    parser.add_argument('--init_file', metavar='s', type=str, default="None", help='Init image')
    parser.add_argument('--write_labels', action='store_true', default=False, help='Write class labels to images')
    parser.add_argument('--output_dir', metavar='b', type=str, default=".", help='Output directory for saving results')
    parser.add_argument('--net_weights', metavar='b', type=str, default=settings.encoder_weights, help='Weights of the net being visualized')
    parser.add_argument('--net_definition', metavar='b', type=str, default=settings.encoder_definition, help='Definition of the net being visualized')

    args = parser.parse_args()

    # Default to constant learning rate
    if args.lr_end < 0:
        args.lr_end = args.lr

    # summary
    print "-------------"
    print " units: %s    xy: %s" % (args.units, args.xy)
    print " n_iters: %s" % args.n_iters
    print " reset_every: %s" % args.reset_every
    print " save_every: %s" % args.save_every
    print " threshold: %s" % args.threshold

    print " epsilon1: %s" % args.epsilon1
    print " epsilon2: %s" % args.epsilon2
    print " epsilon3: %s" % args.epsilon3
    print " epsilon4: %s" % args.epsilon4

    print " start learning rate: %s" % args.lr
    print " end learning rate: %s" % args.lr_end
    print " seed: %s" % args.seed
    print " opt_layer: %s" % args.opt_layer
    print " act_layer: %s" % args.act_layer
    print " init_file: %s" % args.init_file
    print "-------------"
    print " output dir: %s" % args.output_dir
    print " net weights: %s" % args.net_weights
    print " net definition: %s" % args.net_definition
    print "-------------"

    # encoder and generator for images 
    encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)
    generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)

    # condition network, here an image classification net
    net = caffe.Classifier(args.net_definition, args.net_weights,
                             mean = np.float32([104.0, 117.0, 123.0]), # ImageNet mean
                             channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    # Fix the seed
    np.random.seed(args.seed)

    # Sampler for class-conditional generation
    sampler = ClassConditionalSampler()
    inpainting = None

    if args.init_file != "None":

        # Pre-compute masks if we want to perform inpainting 
        if args.epsilon4 > 0:
            mask, neg = util.get_mask()

        # Get the code for the masked image
        start_code, start_image = get_code(encoder=encoder, path=args.init_file, layer=args.opt_layer, mask=neg)

        # Package settings for in-painting experiments
        if args.epsilon4 > 0:
            inpainting = {
                "mask"      : mask,
                "mask_neg"  : neg,
                "image"     : start_image
            }

        print "Loaded init code: ", start_code.shape
    else:
        # shape of the code being optimized
        shape = generator.blobs[settings.generator_in_layer].data.shape
        start_code = np.random.normal(0, 1, shape)
        print ">>", np.min(start_code), np.max(start_code)

    # Separate the dash-separated list of units into numbers
    conditions = [ { "unit": int(u), "xy": args.xy } for u in args.units.split("_") ]       
    
    # Optimize a code via gradient ascent
    output_image, list_samples = sampler.sampling( condition_net=net, image_encoder=encoder, image_generator=generator, 
                        gen_in_layer=settings.generator_in_layer, gen_out_layer=settings.generator_out_layer, start_code=start_code, 
                        n_iters=args.n_iters, lr=args.lr, lr_end=args.lr_end, threshold=args.threshold, 
                        layer=args.act_layer, conditions=conditions,
                        epsilon1=args.epsilon1, epsilon2=args.epsilon2, epsilon3=args.epsilon3, epsilon4=args.epsilon4,
                        inpainting=inpainting,
                        output_dir=args.output_dir, 
                        reset_every=args.reset_every, save_every=args.save_every)

    # Output image
    filename = "%s/%s_%04d_%04d_%s_h_%s_%s_%s_%s__%s.jpg" % (
            args.output_dir,
            args.act_layer, 
            conditions[0]["unit"],
            args.n_iters,
            args.lr,
            str(args.epsilon1),
            str(args.epsilon2),
            str(args.epsilon3),
            str(args.epsilon4),
            args.seed
        )

    if inpainting != None:
        output_image = util.stitch(start_image, output_image) 

    # Save the final image
    util.save_image(output_image, filename)
    print "%s/%s" % (os.getcwd(), filename)

    # Write labels to images
    print "Saving images..."
    for p in list_samples:
        img, name, label = p
        util.save_image(img, name)
        if args.write_labels:
            util.write_label_to_img(name, label)

if __name__ == '__main__':
    main()
