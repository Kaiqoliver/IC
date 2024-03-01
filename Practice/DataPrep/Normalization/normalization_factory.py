from .Strategies.normalization_per_channel_RGB_strategy import NormalizationPerChannelRGBStrategy

class NormalizationFactory:
    def __init__(self):
        self.strategy_factory = {
            "per_channel_RGB" : NormalizationPerChannelRGBStrategy()
        }
    
    def get_strategy_instance(self, name):
        return(self.strategy_factory[name])