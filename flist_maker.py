# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:33:45 2019

@author: nadim
"""

#!/usr/bin/python

import os
import fnmatch

########################################################################################################################
# Change directories here
########################################################################################################################

root_train = '/home/nadim/Code_Ground/places2_hr/train_large'
root_val = '/home/nadim/Code_Ground/places2_hr/val_large'

train_filename = '/home/nadim/Code_Ground/untitled_01/flist/train.txt'
validation_filename = '/home/nadim/Code_Ground/untitled_01/flist/eval.txt'

########################################################################################################################
# Image Lister
########################################################################################################################

def dataset_files(rootdir, pattern):
    """Returns a list of all image files in the given directory"""

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches

########################################################################################################################
# If image format is not 'jpg', change accordingly.
########################################################################################################################
train = dataset_files(root_train, '*.jpg')
validation = dataset_files(root_val, '*.jpg')


########################################################################################################################
# Filename maker
########################################################################################################################
#
if not os.path.exists(train_filename):
    os.mknod(train_filename)
#
if not os.path.exists(validation_filename):
    os.mknod(validation_filename)


########################################################################################################################
# Write to file
########################################################################################################################

fo = open(train_filename, "w")
fo.write("\n".join(train))
fo.close()
#
fo = open(validation_filename, "w")
fo.write("\n".join(validation))
fo.close()

