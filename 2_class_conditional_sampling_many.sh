#/bin/bash
#
# Anh Nguyen <anh.ng8@gmail.com>
# 2016

# Take in epsilon1
if [ "$#" -ne "1" ]; then
  echo "Provide epsilon1 e.g. 1e-5"
  exit 1
fi

opt_layer=fc6
act_layer=fc8
list_units="946 629" 

xy=0              # Spatial position for conv layers, for fc layers: xy = 0
n_iters=100       # For each unit, for N iterations
reset_every=0     # For diversity, reset the code to random every N iterations. 0 to disable resetting.
save_every=1      # Save a sample every N iterations
lr=1 
lr_end=1          # Linearly decay toward this ending lr (e.g. for decaying toward 0, set lr_end = 1e-10)
threshold=0       # Filter out samples below this threshold e.g. 0.98

# -----------------------------------------------
# Multipliers in the update rule Eq.11 in the paper
# -----------------------------------------------
epsilon1=${1}       # prior
epsilon2=1        # condition
epsilon3=1e-17    # noise
# -----------------------------------------------

init_file="None"    # Start from a random code

# Condition net
net_weights="nets/caffenet/bvlc_reference_caffenet.caffemodel"
net_definition="nets/caffenet/caffenet.prototxt"
#-----------------------

# Make a list of units
needle=" "
n_units=$(grep -o "$needle" <<< "$list_units" | wc -l)
units=${list_units// /_}

# Output dir
output_dir="output/${act_layer}_chain_${units}_eps1_${epsilon1}_eps3_${epsilon3}"
mkdir -p ${output_dir}


# Directory to store samples
if [ "${save_every}" -gt "0" ]; then
    sample_dir=${output_dir}/samples
    rm -rf ${sample_dir} 
    mkdir -p ${sample_dir} 
fi

unit_pad=`printf "%04d" ${unit}`

for seed in {0..0}; do

    python ./sampling_class.py \
        --act_layer ${act_layer} \
        --opt_layer ${opt_layer} \
        --units ${units} \
        --xy ${xy} \
        --n_iters ${n_iters} \
        --save_every ${save_every} \
        --reset_every ${reset_every} \
        --lr ${lr} \
        --lr_end ${lr_end} \
        --seed ${seed} \
        --output_dir ${output_dir} \
        --init_file ${init_file} \
        --epsilon1 ${epsilon1} \
        --epsilon2 ${epsilon2} \
        --epsilon3 ${epsilon3} \
        --threshold ${threshold} \
        --write_labels \
        --net_weights ${net_weights} \
        --net_definition ${net_definition} \

    # Save the samples
    if [ "${save_every}" -gt "0" ]; then

        f_chain=${output_dir}/chain_${units}_hx_${epsilon1}_noise_${epsilon3}__${seed}.jpg

        # Make a montage of intermediate samples
        echo "Making a collage..."
        montage ${sample_dir}/*.jpg -tile 10x -geometry +1+1 ${f_chain}
        readlink -f ${f_chain}

        echo "Making a gif..."
        convert ${sample_dir}/*.jpg -delay 5 -loop 0 ${f_chain}.gif     
        readlink -f ${f_chain}.gif
    fi
done
