from model import *
from metrics import *
from data import *
#from postproc import *
from keras.models import load_model
#import matplotlib.pyplot as plt
import skimage.io
import cv2
import json
import sys
import os
from smooth_tiled_predictions import predict_img_with_smooth_windowing


# To avoid TF bugs with GPU memory
if True:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4 # todo: actually it should be configured
    session = tf.Session(config=config)


def to_rgb3(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    return np.asarray(np.dstack((im, im*0, im*0)), dtype=np.uint8)



def predict(orthophoto_filename, model, smoothPatches=True, debug = False):
    print('Predicting ...')
    # Since model was trained by images with switched R and B channels here we need switch.
    # OpenCV switch it by default because it use BGR channels formation
    ximg = cv2.imread(orthophoto_filename)[:,:,:3]
    
    ximg = cv2.resize(ximg, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    print('Source image shape: {0}, type: {1}'.format(ximg.shape, ximg.dtype))

    if smoothPatches==True:
        # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
        # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
        prob_field = predict_img_with_smooth_windowing(
            ximg,
            window_size=96,  # Should be more than 96(training shape). But huge size will down your GPU and PC. Test shows 96(as wnd size) is the best
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=1,
            pred_func=(
                lambda img_batch_subdiv: model.predict(img_batch_subdiv)
            )
        )
    else:    
        # Tile orthophoto images
        cropSize = 1024 # should be pow of 2, and greater than source image size used for training, sure - eq/less than img size
        overlapCropSize = cropSize // 4 * 3
        w0,w1,h0,h1 = getTiledBoxx(ximg.shape, tile_size=(cropSize, cropSize), offset=(overlapCropSize, overlapCropSize))
        print('Number of tiles: {0}'.format(len(w0)))
        ximg_tiled = []
        for i in range(len(w0)):
            ximg_tiled.append(ximg[h0[i]:h1[i], w0[i]:w1[i]])

        # Convert image tiles to required data format
        ximg_tiled = np.asarray(ximg_tiled, dtype=np.float32)
        print('Array prepared for prediction shape: {0}, type: {1}'.format(ximg_tiled.shape, ximg_tiled.dtype))

        # To avoid memory overload, predict samples partialy
        tilesDoneNb = 0
        tilesDoneSetp = 2
        preds_train = np.empty(shape=(0, cropSize, cropSize, 1))
        while tilesDoneNb < ximg_tiled.shape[0]:
            temp = model.predict(ximg_tiled[tilesDoneNb:min(tilesDoneNb+tilesDoneSetp, ximg_tiled.shape[0])], verbose=0)
            preds_train = np.append(preds_train, temp, axis=0)
            tilesDoneNb = tilesDoneNb + tilesDoneSetp

        print('Predicted result shape: {0}, type: {1}'.format(preds_train.shape, preds_train.dtype))

        # Accum tiled results to whole image
        ximg_accum_gray = np.zeros(shape=(ximg.shape[0],ximg.shape[1],1), dtype=np.uint8)
        prob_field = np.zeros(shape=(ximg.shape[0],ximg.shape[1],1), dtype=np.float32)
        for i in range(len(w0)):
            #prob_field[h0[i]:h1[i], w0[i]:w1[i]] = preds_train[i]
            prob_field[h0[i]:h1[i], w0[i]:w1[i]] = np.fmax(prob_field[h0[i]:h1[i], w0[i]:w1[i]], preds_train[i])

    # result shape [h,w]
    return (prob_field[:,:,0] * 255).astype(np.uint8)

if __name__ == "__main__":
    # Create model without specific input size
    model = create_model(img_hw=None, input_channels=3, dropout_value=0.0)
    # If use atrous model, get it by create_model_atrous() and load from solution ../solutions/bce_logiou_thorbn/nadam_do02_posRatio-1_batch32
    checkpointFilename = FindBestCheckpoint('../solutions/bce_logiou/nadam_do02_posRatio-1_batch32', 'weights-')
    if len(checkpointFilename)==0:
        print('ERROR: Has no any checkpoint by specified path')
        exit(-1)
    model.load_weights(checkpointFilename)
 
    debug = True
    smoothPatches = True # If true: 8 times slower but much better result
    folderPath = sys.argv[1]
    file_names = [fn for fn in os.listdir(folderPath)]
    for fname in file_names:
        fullpath = os.path.join(folderPath, fname)
        prob_field_u8c1 = predict(fullpath, model, smoothPatches=smoothPatches, debug=debug)
        propbpath = os.path.join(folderPath, 'prob_' + fname)
        markpath = os.path.join(folderPath, 'mark_' + fname)

        skimage.io.imsave(propbpath, np.dstack((prob_field_u8c1, prob_field_u8c1, prob_field_u8c1)))
        #
        newX = prob_field_u8c1.shape[1]
        newY = prob_field_u8c1.shape[0]
        img = cv2.imread(fullpath)[:,:,:3]
        img = cv2.resize(img, (int(newX),int(newY)), interpolation=cv2.INTER_CUBIC)

        threshold = 0.7
        mask = prob_field_u8c1 > threshold*255
        mask = mask.astype(np.uint8)
        if cv2.__version__.startswith("3"):
            im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if False:
            cv2.drawContours(img, contours, -1, (0,0,255), 3)
        else:
            minArea = newX*newY
            maxArea = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                minArea = min(minArea, area)
                maxArea = max(maxArea, area)

            #img[mask] = (0,0,255)
            eps = 0.0001
            for cnt in contours:
                area = cv2.contourArea(cnt)
                rarea = (area - minArea + eps) / (maxArea - minArea + eps)
                x,y,w,h = cv2.boundingRect(cnt)
                size = max(w,h)
                cx = x + w//2
                cy = y + h//2
                x = cx - size//2
                y = cy - size//2
                x = max(x,0)
                y = max(y,0)
                x2 = cx + size//2
                y2 = cy + size//2
                x2 = min(x2,newX-1)
                y2 = min(y2,newY-1)
                w = x2 - x
                h = y2 - y
                cv2.rectangle(img,(x,y),(x+w,y+h),(255*(1-rarea),0,255*rarea),2)

        cv2.imwrite(markpath, img)
