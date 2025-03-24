from sklearn.cross_decomposition import PLSRegression

from models.base_model import BaseExperimentArgs, BaseExperimentModel

class PLSRModelArgs(BaseExperimentArgs):
    n_components: int = 1
     
class PLSRModel(BaseExperimentModel):
    def __init__(self, config: PLSRModelArgs):
        super().__init__()
        self.n_components = config.n_components
        
    def train(self, input_train, target_train):
        self.model = PLSRegression(
            n_components=self.n_components,
            scale=False)
        
        self.model.fit(input_train, target_train)
        return
    
    @classmethod
    def get_args_model(cls):
        return PLSRModelArgs