from data import *
from model import *
from metrics import *
from keras.optimizers import Adam
import sys

def sortCheckpoints(checkpoint):
    def get(x):
        try:
            val_loss,val_mean_iou = parseCheckpointFilename(x)
        except:
            val_loss = sys.float_info.max
        return val_loss
    # First will be min val_loss
    checkpoint = sorted(checkpoint, key=lambda x: get(x), reverse=False)
    return checkpoint

rootpath = '../solutions/shuffle/aug/fordel'
#rootpath = '../solutions/simulate'
checkpoints = getCheckpoints(rootpath, 'weights-')

checkpoints = sortCheckpoints(checkpoints)

print(checkpoints)

#print(checkpoints)
img_hw = 96
samplesNb = prepareData('../', '../test', output_shape=(img_hw,img_hw), minPositiveRatio=-1, augNb=1, foldNb=1, foldInd=0)


batch_size = 32
generator = get_generator('../test/.ds_val', {}, img_hw, batch_size, use_shuffle=False, loc_seed=42)

model = create_model(img_hw=img_hw, input_channels=3, dropout_value=0.0)

# 
for checkPath in checkpoints:
    model.load_weights(checkPath)

    # Set some model compile parameters
    optimizer = Adam(lr=0.0001)
    #optimizer = 'adadelta'
    #loss      = bce_dice_loss
    loss = 'binary_crossentropy'
    #metrics   = [mean_iou]
    metrics = ['accuracy', mean_iou]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    score = model.evaluate_generator(generator, steps=samplesNb//batch_size, verbose=1)
    #score = model.predict_generator(generator, verbose=1, steps=samplesNb//batch_size)
    #print(score.shape)
    print('Checkpoint: ', checkPath)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
