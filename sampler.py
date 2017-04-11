#!/usr/bin/env python
'''
Anh Nguyen <anh.ng8@gmail.com>
2017
'''

import os, sys
os.environ['GLOG_minloglevel'] = '2'    # suprress Caffe verbose prints

import settings
sys.path.insert(0, settings.caffe_root)
import caffe

import numpy as np
from numpy.linalg import norm
import scipy.misc, scipy.io
import util

class Sampler(object):
    
    def backward_from_x_to_h(self, generator, diff, start, end):
        '''
        Backpropagate the gradient from the image (start) back to the latent space (end) of the generator network.
        '''
        dst = generator.blobs[end]

        dst.diff[...] = diff
        generator.backward(start=end)
        g = generator.blobs[start].diff.copy()

        dst.diff.fill(0.)   # reset objective after each step

        return g

    def h_autoencoder_grad(self, h, encoder, decoder, gen_out_layer, topleft, inpainting):
        '''
        Compute the gradient of the energy of P(input) wrt input, which is given by decode(encode(input))-input {see Alain & Bengio, 2014}.
        Specifically, we compute E(G(h)) - h.
        Note: this is an "upside down" auto-encoder for h that goes h -> x -> h with G modeling h -> x and E modeling x -> h.
        '''

        generated = encoder.forward(feat=h)
        x = encoder.blobs[gen_out_layer].data.copy()    # 256x256
        
        # Crop from 256x256 to 227x227
        image_size = decoder.blobs['data'].shape    # (1, 3, 227, 227)
        cropped_x = x[:,:,topleft[0]:topleft[0]+image_size[2], topleft[1]:topleft[1]+image_size[3]]

        # Mask the image when inpainting
        if inpainting is not None:
            cropped_x = util.apply_mask(img=cropped_x, mask=inpainting['mask'], context=inpainting['image'])

        # Push this 227x227 image through net
        decoder.forward(data=cropped_x)
        code = decoder.blobs['fc6'].data

        g = code - h

        return g


    def sampling( self, condition_net, image_encoder, image_generator, 
                gen_in_layer, gen_out_layer, start_code, 
                n_iters, lr, lr_end, threshold, 
                layer, conditions, #units=None, xy=0, 
                epsilon1=1, epsilon2=1, epsilon3=1e-10,
                inpainting=None, # in-painting args
                output_dir=None, reset_every=0, save_every=1):

        # Get the input and output sizes
        image_shape = condition_net.blobs['data'].data.shape
        generator_output_shape = image_generator.blobs[gen_out_layer].data.shape
        encoder_input_shape = image_encoder.blobs['data'].data.shape

        # Calculate the difference between the input image of the condition net 
        # and the output image from the generator
        image_size = util.get_image_size(image_shape)
        generator_output_size = util.get_image_size(generator_output_shape)
        encoder_input_size = util.get_image_size(encoder_input_shape)

        # The top left offset to crop the output image to get a 227x227 image
        topleft = util.compute_topleft(image_size, generator_output_size)
        topleft_DAE = util.compute_topleft(encoder_input_size, generator_output_size)

        src = image_generator.blobs[gen_in_layer]     # the input feature layer of the generator
        
        # Make sure the layer size and initial vector size match
        assert src.data.shape == start_code.shape

        # Variables to store the best sample
        last_xx = np.zeros(image_shape)    # best image
        last_prob = -sys.maxint                 # highest probability 

        h = start_code.copy()

        condition_idx = 0 
        list_samples = []
        i = 0

        while True:

            step_size = lr + ((lr_end - lr) * i) / n_iters
            condition = conditions[condition_idx]  # Select a class

            # 1. Compute the epsilon1 term ---
            # compute gradient d log(p(h)) / dh per DAE results in Alain & Bengio 2014
            d_prior = self.h_autoencoder_grad(h=h, encoder=image_generator, decoder=image_encoder, gen_out_layer=gen_out_layer, topleft=topleft_DAE, inpainting=inpainting)

            # 2. Compute the epsilon2 term ---
            # Push the code through the generator to get an image x
            image_generator.blobs["feat"].data[:] = h
            generated = image_generator.forward()
            x = generated[gen_out_layer].copy()       # 256x256

            # Crop from 256x256 to 227x227
            cropped_x = x[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]
            cropped_x_copy = cropped_x.copy()
            
            if inpainting is not None:
                cropped_x = util.apply_mask(img=cropped_x, mask=inpainting['mask'], context=inpainting['image'])

            # Forward pass the image x to the condition net up to an unit k at the given layer
            # Backprop the gradient through the condition net to the image layer to get a gradient image 
            d_condition_x, prob, info = self.forward_backward_from_x_to_condition(net=condition_net, end=layer, image=cropped_x, condition=condition) 

            if inpainting is not None:
                # Mask out the class gradient image
                d_condition_x[:] *= inpainting["mask"]

                # An additional objective for matching the context image
                d_context_x256 = np.zeros_like(x.copy())
                d_context_x256[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = (inpainting["image"] - cropped_x_copy) * inpainting["mask_neg"]
                d_context_h = self.backward_from_x_to_h(generator=image_generator, diff=d_context_x256, start=gen_in_layer, end=gen_out_layer)

            # Put the gradient back in the 256x256 format 
            d_condition_x256 = np.zeros_like(x)
            d_condition_x256[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = d_condition_x.copy()

            # Backpropagate the above gradient all the way to h (through generator)
            # This gradient 'd_condition' is d log(p(y|h)) / dh (the epsilon2 term in Eq. 11 in the paper)
            d_condition = self.backward_from_x_to_h(generator=image_generator, diff=d_condition_x256, start=gen_in_layer, end=gen_out_layer)

            self.print_progress(i, info, condition, prob, d_condition)

            # 3. Compute the epsilon3 term ---
            noise = np.zeros_like(h)
            if epsilon3 > 0:
                noise = np.random.normal(0, epsilon3, h.shape)  # Gaussian noise

            # Update h according to Eq.11 in the paper 
            d_h = epsilon1 * d_prior + epsilon2 * d_condition + noise

            # Plus the optional epsilon4 for matching the context region when in-painting
            if inpainting is not None:
                d_h += inpainting["epsilon4"] * d_context_h 

            h += step_size/np.abs(d_h).mean() * d_h

            h = np.clip(h, a_min=0, a_max=30)   # Keep the code within a realistic range

            # Reset the code every N iters (for diversity when running a long sampling chain)
            if reset_every > 0 and i % reset_every == 0 and i > 0: 
                h = np.random.normal(0, 1, h.shape)

                # Experimental: For sample diversity, it's a good idea to randomly pick epsilon1 as well
                epsilon1 = np.random.uniform(low=1e-6, high=1e-2)

            # Save every sample
            last_xx = cropped_x.copy()
            last_prob = prob

            # Filter samples based on threshold or every N iterations
            if save_every > 0 and i % save_every == 0 and prob > threshold:
                name = "%s/samples/%05d.jpg" % (output_dir, i)

                label = self.get_label(condition)
                list_samples.append( (last_xx.copy(), name, label) ) 

            # Stop if grad is 0
            if norm(d_h) == 0:
                print " d_h is 0"
                break

            # Randomly sample a class every N iterations
            if i > 0 and i % n_iters == 0:
                condition_idx += 1

                if condition_idx == len(conditions):
                    break

            i += 1  # Next iter

        # returning the last sample
        print "-------------------------"
        print "Last sample: prob [%s] " % last_prob

        return last_xx, list_samples
