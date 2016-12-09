# Set this to the path to Caffe installation on your system
caffe_root = "/path/to/your/caffe/python"
gpu = True

# -------------------------------------
# The following are hard-coded and hardly change unless we change to use a different generator.
# -------------------------------------
# Generator G 
generator_weights = "nets/generator/noiseless/generator.caffemodel"
generator_definition = "nets/generator/noiseless/generator.prototxt"

# input / output layers in the generator prototxt
generator_in_layer = "feat"
generator_out_layer = "deconv0"

# Encoder E
encoder_weights = "nets/caffenet/bvlc_reference_caffenet.caffemodel"
encoder_definition = "nets/caffenet/caffenet.prototxt"

# Text files
synset_file = "misc/synset_words.txt"
vocab_file = "misc/vocabulary.txt"
