# use lableme to add fake lable on picture to point out which part is hair, which part is skin, cloth, eyes and others
# lableme produce json file, it record different part of body via polygon 
# we need to judge each point whether it is in one of the polygon and classfy them to different part 
# one way is to divide to five matrices, if one pixel is in hair area, change it to 1
# send hair mask and origin picture to unet, make it return preview mask
# the skin mask should cover the hair mask

