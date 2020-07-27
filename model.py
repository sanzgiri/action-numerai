import numerox as nx
import joblib
import os

from sklearn.linear_model import LinearRegression

# You can find your model_id at https://numer.ai/models
# model_id = os.environ["MODEL_ID"]

# define a model that can be trained separately and saved
class LinearModel(nx.Model):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.model = LinearRegression()
        self.model_id = model_id

    def fit(self, dfit, tournament):
        self.model.fit(dfit.x, dfit.y[tournament])

    def fit_predict(self, dfit, dpre, tournament):
        # fit is done separately in `.fit()`

        # predict
        yhat = self.model.predict(dpre.x)

        # return predictions along with the original ids
        return dpre.ids, yhat

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)
