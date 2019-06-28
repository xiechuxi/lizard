# import comet_ml in the top of your file
from comet_ml import Experiment
    
##################################################################################################################################
# Setup reproducable results at the start of run-script(todo: before comet_ml?)
#
# src: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

import random as rn
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

import tensorflow as tf
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)


if True: # Use GPU?
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    # To fix GPU memory issue
    # https://github.com/tensorflow/tensorflow/issues/6698#issuecomment-297179317
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.06 # 0.06 for 16GB GPU is enough
    session_conf.gpu_options.allow_growth = False # True ?
else:
    session_conf = tf.ConfigProto() # Use all CPU cores

from keras import backend as K


sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#
##################################################################################################################################

# Runtime custom callbacks
#%% https://github.com/deepsense-ai/intel-ai-webinar-neural-networks/blob/master/live_loss_plot.py
# Fixed code to enable non-flat loss plots on keras model.fit_generator()
import matplotlib
matplotlib.use('Agg') # Disable TclTk because it sometime crash training!

# from comet_ml import Experiment
from data import *
from model import *
from metrics import *
import os.path
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

from keras.callbacks import Callback
from IPython.display import clear_output



def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x

class PlotLosses(Callback):
    def __init__(self, imgfile, figsize=None):
        super(PlotLosses, self).__init__()
        self.figsize = figsize
        self.imgfile = imgfile

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs.copy())

        clear_output(wait=True)
        fig = plt.figure(figsize=self.figsize)
        
        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)
            
            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label="training")
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('epoch')
            plt.legend(loc='center left')
        
        plt.tight_layout()
        fig.savefig(self.imgfile)
        plt.draw() # draw the plot. Actually it could crash training. To avoid crash add 'matplotlib.use('Agg')' at the start
        plt.pause(2) # show it for N seconds
        plt.close(fig)

def train(srcdir, param):
    img_hw = 96
    foldNb = 4
    solutionsFolder = os.path.join(srcdir, 'solutions')
    solutionFolder = os.path.join(solutionsFolder, param['solution'])
    projectFolder = os.path.join(solutionFolder, param['name'])
    if os.path.isdir(projectFolder)==True:
        if query_yes_no('Project folder {0} already exist. Do you want to continue fitting'.format(projectFolder), default="no")==False:
            exit()
 
    validation_split = 1.0 / foldNb # Fraction of images reserved for validation. 0->AllTrain, 1->AllTest
    for foldInd in range(foldNb):
        outputFoler = os.path.join(projectFolder, 'fold_' + str(foldInd))
        if os.path.isdir(outputFoler)==True:
            print('Fold {} already exist. Skip from fitting'.format(outputFoler))
            continue
        recreateFolder(outputFoler)

        samplesNb,inp_chNb = prepareData(srcdir,
                                        outputFoler,
                                        (img_hw,img_hw),                            # Crop source samples by specified size
                                        minPositiveRatio=param['minPositiveRatio'], # Specify samples' quality
                                        augNb=param['srcAugMult'],                  # Multiply source dataset by # augmented samples
                                        foldNb=foldNb,
                                        foldInd=foldInd)

        model = create_model(img_hw=img_hw, input_channels=inp_chNb, dropout_value=param['dropout_value'])
        # Set the best checkpoint if parent folder specified(relatively to source dir)
        if len(param['parent']) > 0:
            parentFolder = os.path.join(solutionsFolder, param['parent'])
            bestWeightsPath = FindBestCheckpoint(parentFolder, 'weights-')
            if len(bestWeightsPath) > 0:
                model.load_weights(bestWeightsPath)
                print('Model initialized by checkpoint {0}'.format(bestWeightsPath))
                with open(os.path.join(projectFolder, 'parent.txt'), "w") as text_file:
                    print('Parent project: {0}'.format(bestWeightsPath), file=text_file)
            else:
                print('ERROR: Specified parent checkpoint does not exist. To not start process from the scratch process was interrupted')
                exit(-1)


        # Set some model compile parameters
        #optimizer = Adam(lr=0.0001)
        #optimizer = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
        #optimizer = 'adagrad'
        #optimizer = 'nadam' # get faster best result than adam
        optimizer = param['optimizer']
        #optimizer = 'adadelta'
        #loss      = bce_dice_loss
        #loss = 'binary_crossentropy'
        loss =  custom_loss
        #metrics   = [mean_iou]
        #metrics = ['accuracy', mean_iou]
        metrics = ['accuracy', mean_iou]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # PNG-files processed in Windows & Ubuntu
        plot_losses = PlotLosses(imgfile=os.path.join(outputFoler, 'log.png'), figsize=(8, 4))

        batch_size = param['batch_size']
        epochs=30000

        # Prepare callbacks
        filepath = os.path.join(outputFoler,
                                'weights-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-val_acc{val_acc:.5f}-val_mean_iou{val_mean_iou:.5f}.h5')
        checkpointer = ModelCheckpoint(filepath, 
                                       monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='max')
        trainInterrupter = EarlyStopping(monitor='val_acc',
                                         min_delta=0,
                                         patience=param['patience'],
                                         verbose=0, mode='max')


        # Image data generator distortion options for training set
        data_gen_args = dict(rotation_range=param['rotation_range'],
                             width_shift_range=param['shift_range'],
                             height_shift_range=param['shift_range'],
                             shear_range=param['shear_range'],
                             zoom_range=param['zoom_range'],
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect' # constant, reflect
                             )

        # Prepare generators
        train_generator = get_generator(os.path.join(outputFoler,'.ds_train'), 
                                        data_gen_args,
                                        img_hw,
                                        batch_size,
                                        use_shuffle=True,
                                        loc_seed=42)
        test_generator  = get_generator(os.path.join(outputFoler,'.ds_val'),
                                        {},
                                        img_hw,
                                        batch_size,
                                        use_shuffle=True,
                                        loc_seed=42)

        if False: # to see what exactly generated
            cind = 0
            for ig in test_generator:
                imgs = ig[0]
                masks = ig[1]
                for im in range(imgs.shape[0]):
                    img = imgs[im,:,:,:]
                    mask = masks[im,:,:,:]
                    img = img + mask*127
                    cv2.imwrite('img_' + str(cind) + '.png', img)
                    cind = cind + 1
                    if cind >= 140:
                        exit(-1)

        try:
            # Each new fit should create new Experiment
            #
            # Add the following code anywhere in your machine learning file
            # Rest API key defined in the HOME directory (for more details see: https://www.comet.ml/docs/python-sdk/advanced/)
            experiment = Experiment(project_name="lizunet", workspace="strayos")

            # Start training
            model.fit_generator(train_generator, 
                                validation_data=test_generator, 
                                validation_steps=samplesNb*validation_split/batch_size, 
                                steps_per_epoch=samplesNb*(1-validation_split)/batch_size, 
                                epochs=epochs,
                                callbacks=[plot_losses,checkpointer,trainInterrupter],
                                verbose=1,
                                shuffle=True)
        except Exception as err:
            err_str = str(err)
            print('ERROR:', err_str)

        #K.clear_session()
        #del model

    # Inform about project just finished
    print('Training of project {0} is finished'.format(projectFolder))

if __name__ == "__main__":
    params = []
    params.append({
             'solution': 'bce_logiou/adam-001_do05_posRatio0_batch16',
             'name': 'x10_360_02_1_03',
             'parent': '',
             'optimizer': Adam(lr=0.001),
             'dropout_value': 0.5,
             'minPositiveRatio': 0.0,
             'batch_size': 16,
             'srcAugMult': 10,
             'patience': 200,
             'rotation_range': 360.0,
             'shift_range': 0.2,
             'shear_range': 1.0,
             'zoom_range': 0.3
            })
    params.append({
             'solution': 'bce_logiou/nadam_do02_posRatio-1_batch32',
             'name': 'x20_360_02_1_03',
             'parent': 'bce_logiou/adam-001_do05_posRatio0_batch16/x10_360_02_1_03',
             'optimizer': 'nadam',
             'dropout_value': 0.2,
             'minPositiveRatio': -1.0,
             'batch_size': 32,
             'srcAugMult': 20,
             'patience': 500,
             'rotation_range': 360.0,
             'shift_range': 0.2,
             'shear_range': 1.0,
             'zoom_range': 0.3
            })

    for param in params:
        train('../', param)