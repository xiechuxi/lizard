import warnings
import glob
import numpy
import os
import re
import math
import numpy as np
import cv2
import shutil
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import random

maskExt = 'mask.png'

_debug = False

class Sample:
    imgFile = ''
    maskFile = ''
    def print(this):
        print("{0},{1}".format(this.imgFile, this.maskFile))

def get_files_by_ext(relevant_path = './', included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']):
    file_names = [fn for fn in os.listdir(relevant_path)
        if any(fn.endswith(ext) for ext in included_extensions)]
    file_names.sort() # To provide reproducible training model
    return file_names

def recreateFolder(fname):
    if os.path.isdir(fname)==True:
        shutil.rmtree(fname)
    os.makedirs(fname)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def getTiledBoxx(img_shape, tile_size, offset):
    aw0 = []
    aw1 = []
    ah0 = []
    ah1 = []
    for i in range(int(math.ceil(1 + (img_shape[0] - tile_size[1])/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(1 + (img_shape[1] - tile_size[0])/(offset[0] * 1.0)))):
            h1 = min(offset[1]*i+tile_size[1], img_shape[0])
            h0 = max(0, h1 - tile_size[1])
            w1 = min(offset[0]*j+tile_size[0], img_shape[1])
            w0 = max(0, w1 - tile_size[0])
            aw0.append(w0)
            aw1.append(w1)
            ah0.append(h0)
            ah1.append(h1)
    return aw0,aw1,ah0,ah1

def get_generator(srcdir, data_gen_args, img_hw, batch_size, use_shuffle=True, loc_seed=42):
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args, rescale=1./255.)

    image_generator = image_datagen.flow_from_directory(
        os.path.join(srcdir, 'imgs'),
        class_mode=None, # this means our generator will only yield batches of data, no labels
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=use_shuffle,
        target_size=(img_hw, img_hw),
        seed=loc_seed)
    mask_generator = mask_datagen.flow_from_directory(
        os.path.join(srcdir, 'masks'),
        class_mode=None, # this means our generator will only yield batches of data, no labels
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=use_shuffle,
        target_size=(img_hw, img_hw),
        seed=loc_seed)

    # combine generators into one which yields image and masks
    generator = zip(image_generator, mask_generator)

    return generator

# Prepare training data
def prepareData(srcdir, outputFoler, output_shape, minPositiveRatio, augNb, foldNb, foldInd):

    imgPath = os.path.join(srcdir, 'imgs')
    maskPath = os.path.join(srcdir, 'masks')

    imgFiles = get_files_by_ext(imgPath)
    maskFiles = get_files_by_ext(maskPath, [maskExt])

    maskFilesWoExt = [id.split('.')[0] for id in maskFiles]
    samples = []
    for imgFile in imgFiles:
        imgFileWoE = os.path.splitext(imgFile)[0]
        if imgFileWoE in maskFilesWoExt :
            sample = Sample()
            sample.imgFile = imgPath + '/' + imgFile
            sample.maskFile = maskPath + '/' + imgFileWoE + '.' + maskExt
            samples.append(sample)
    random.Random(42).shuffle(samples)


    # Prepare folders for cropped images
    # Since ImageGenerator will process them, mark subfolders as the single same classes('0')
    cropImgPath_train = os.path.join(outputFoler,'.ds_train/imgs/0')
    cropMaskPath_train = os.path.join(outputFoler,'.ds_train/masks/0')
    cropImgPath_val = os.path.join(outputFoler,'.ds_val/imgs/0')
    cropMaskPath_val = os.path.join(outputFoler,'.ds_val/masks/0')
    cropImgPath_temp = os.path.join(outputFoler,'.ds_temp/imgs/0')
    cropMaskPath_temp = os.path.join(outputFoler,'.ds_temp/masks/0')
    cropImgPath_tempAug = os.path.join(outputFoler,'.ds_tempAug/imgs/0')
    cropMaskPath_tempAug = os.path.join(outputFoler,'.ds_tempAug/masks/0')
    recreateFolder(cropImgPath_train)
    recreateFolder(cropMaskPath_train)
    recreateFolder(cropImgPath_val)
    recreateFolder(cropMaskPath_val)
    recreateFolder(cropImgPath_temp)
    recreateFolder(cropMaskPath_temp)
    recreateFolder(cropImgPath_tempAug)
    recreateFolder(cropMaskPath_tempAug)

    inpuChannelsNb = 0
    # We can control minimum percent of positive pixels in sample image
    # To count used images initialize a following variable
    cropImgIndex = 0

    # Crop images by specified size and collect them
    for sample in samples:
        ximg = cv2.imread(sample.imgFile)[:,:,:3]
        yimg = cv2.imread(sample.maskFile, cv2.IMREAD_GRAYSCALE)

        inpuChannelsNb = ximg.shape[2]

        w0,w1,h0,h1 = getTiledBoxx(ximg.shape, output_shape, output_shape)
        for i in range(len(w0)):
            if np.count_nonzero(yimg[h0[i]:h1[i], w0[i]:w1[i]]) <= (h1[i]-h0[i])*(w1[i]-w0[i])*minPositiveRatio:
                continue

            # Store copped images to temporary folder
            # Disable warning during skimage saving(pure white/black images)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv2.imwrite(os.path.join(cropImgPath_temp, str(cropImgIndex)+'.png'), (ximg[h0[i]:h1[i], w0[i]:w1[i]]).astype(np.uint8))
                cv2.imwrite(os.path.join(cropMaskPath_temp, str(cropImgIndex)+'.png'), (yimg[h0[i]:h1[i], w0[i]:w1[i]]).astype(np.uint8))

            cropImgIndex = cropImgIndex + 1


    # Augmentate the cropped images before dividing them by KFold
    if augNb > 0:
        data_gen_args = dict(rotation_range    = 360.0, # degreee
                             width_shift_range = 0.2,
                             height_shift_range= 0.2,
                             shear_range       = 2.0, # degreee
                             zoom_range        = 0.2,
                             horizontal_flip   = True,
                             vertical_flip     = True,
                             fill_mode         = 'reflect' # constant, reflect
                             )

        # To not overload memory, use generator partially
        # e.g. to process all samples we will need use generator N times(not once)
        needMemBytes = cropImgIndex * output_shape[0] * output_shape[1] * (inpuChannelsNb + 1)
        availableMemBytes = 2**24 # 16 MB
        genNb = int(math.ceil(needMemBytes / availableMemBytes))
        batch_size = int(math.ceil(cropImgIndex / genNb))
        aug_generator = get_generator(os.path.join(outputFoler,'.ds_temp'),
                                      data_gen_args,
                                      img_hw=output_shape[0],
                                      batch_size=batch_size,
                                      use_shuffle=False, # Do not shuffle to process all samples by \genNb calling
                                      loc_seed=42)
        aug_ind = 0
        cropImgIndex = 0
        for ig in aug_generator: # Since batch size eq to count of img it will iterate whole images by one loop
            imgs = ig[0]
            masks = ig[1]
            for im in range(imgs.shape[0]):
                img = imgs[im,:,:,:]
                mask = masks[im,:,:,0]*255
                # Pay attention here, that augmentation generator returns RGB, but OpenCV stores them as BGR.
                # It means that images will have switched R and B channels. It means that during testing we should
                # read image not as RGB, but BGR(default for OpenCV)
                cv2.imwrite(os.path.join(cropImgPath_tempAug,  str(cropImgIndex) + '.png'), img)
                cv2.imwrite(os.path.join(cropMaskPath_tempAug,  str(cropImgIndex) + '.png'), mask)
                cropImgIndex = cropImgIndex + 1
            aug_ind = aug_ind + 1
            if aug_ind >= (augNb*genNb):
                break
        shutil.rmtree(os.path.join(outputFoler,'.ds_temp'))
        shutil.move(os.path.join(outputFoler,'.ds_tempAug'), os.path.join(outputFoler,'.ds_temp'))


    # Divide images from temporary folder to KFold
    valSamplesNb = cropImgIndex//foldNb
    valInd0 = foldInd*valSamplesNb
    valInd1 = valInd0+valSamplesNb
    file_names = [fn for fn in os.listdir(cropImgPath_temp)]
    file_names.sort() # To provide reproducible training model
    # Since we used not shuffled generator(each aug iteration samples are repeated) 
    # as well as cropped parts was stored as stacked indices we need shuffle result images
    random.Random(42).shuffle(file_names)
    for sind,fnamepath in enumerate(file_names):
        filename = os.path.basename(fnamepath)
        if sind >= valInd0 and sind < valInd1:
            shutil.move(os.path.join(cropImgPath_temp, filename), os.path.join(cropImgPath_val, filename))
            shutil.move(os.path.join(cropMaskPath_temp, filename), os.path.join(cropMaskPath_val, filename))
        else:
            shutil.move(os.path.join(cropImgPath_temp, filename), os.path.join(cropImgPath_train, filename))
            shutil.move(os.path.join(cropMaskPath_temp, filename), os.path.join(cropMaskPath_train, filename))

    # Remove temporary folder
    shutil.rmtree(os.path.join(outputFoler,'.ds_temp'))

    return cropImgIndex, inpuChannelsNb

"""
def getCheckpoints(directory, fileNameStarter):
    checkpoints = []
    for filename in os.listdir(directory):
        if filename.endswith('.h5') and filename.startswith(fileNameStarter): 
            checkpoints.append(filename)
    return checkpoints
"""
def getCheckpoints(dirName, fileNameStarter):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            if entry.startswith('.'):
                print('Checkpoint ignored in folder {0}'.format(fullPath))
            else:
                allFiles = allFiles + getCheckpoints(fullPath, fileNameStarter)
        else:
            if entry.endswith('.h5') and entry.startswith(fileNameStarter): 
                allFiles.append(fullPath)
                
    return allFiles

def parseCheckpointFilename(line):
    mask = re.compile(r"val_loss([-]?[0-9.]+)-val_acc([-]?[0-9.]+)-val_mean_iou([-]?[0-9]+[.]?[0-9]+)")
    #mask = re.compile(r"val_loss([-]?[0-9.]+)")
    #mask = re.compile(r"val_loss([-]?[0-9.]+)-val_mean_iou([-]?[0-9]+[.]?[0-9]+)")
    res = mask.search(os.path.basename(line))
    if res:
        return (float(item) for item in res.groups(0))
    else:
        raise ValueError("Wrong checkpoint filename format: {}".format(line))


def getBestCheckpoint(checkpoints):
    best_val_param = 10.0
    best_index = -1
    for ind, file in enumerate(checkpoints):
        try:
            val_loss,val_acc,val_mean_iou = parseCheckpointFilename(file)
            if val_acc > best_val_param or best_index < 0:
                best_val_param = val_acc
                best_index = ind
        except Exception as err:
            err_str = str(err)
            print('ERROR:', err_str)
    return best_index


def FindBestCheckpoint(rootdir, fileNameStarter):
    checkpoints = getCheckpoints(rootdir, fileNameStarter)
    best_checkpoint_index = getBestCheckpoint(checkpoints)
    if best_checkpoint_index < 0:
        return("")
    val_loss,val_acc,val_mean_iou = parseCheckpointFilename(checkpoints[best_checkpoint_index])
    print('Found best checkpoint: {}'.format(checkpoints[best_checkpoint_index]))
    print('val_loss: {}, val_acc: {}, val_mean_iou: {}'.format(val_loss, val_acc, val_mean_iou))
    return checkpoints[best_checkpoint_index]