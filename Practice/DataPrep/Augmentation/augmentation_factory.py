from .Strategies.augmentation_flip_horizontal_strategy import AugmentationFlipHorizontal

class AugmentationFactory:
    def __init__(self):
        self.strategy_factory = {
            "flip_horizontal" : AugmentationFlipHorizontal()
        }
    
    def get_strategy_instace(self, name):
        return(self.strategy_factory[name])
