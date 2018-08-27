class BaseOptions():

    def initialize(self):
        self.dataroot = '/home/user/Ryan3/dain/noise2noise/Dataset/'
        self.checkpoints_dir = './checkpoints' # models are saved here
        self.name = 'Convolutional_Denoising_AutoEncoder' # Name of the experiment
        self.image_size = 128  
        self.batch_size =5
        self.GPUs = '0,1'

class TrainOptions(BaseOptions):
    
    def __init__(self):

        BaseOptions.initialize(self)
        self.patch_num = 10
        self.phase = 'train'
        self.niter = 1000 # number of iter at starting learning rate
        self.lr = 0.002 # initial learning rate for adam

class TestOptions(BaseOptions):
    
    def __init__(self):

        BaseOptions.initialize(self)
        self.ntest = float('inf') # number of test examples
        self.result_dir = './results/' # saves results here
        self.phase = 'test'  # train,val,test etc
        self.which_epoch = '380'  # which epoch to load? set to latest to use latest cached model

