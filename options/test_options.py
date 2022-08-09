from .base_options import BaseOptions
import os
import utils
from options import config

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):  
        parser = BaseOptions.initialize(self, parser)  # define shared options
    
        parser.set_defaults(outf=os.path.join(config.RESULT_DIR,'debug'))
        
        self.isTrain = False
        return parser
