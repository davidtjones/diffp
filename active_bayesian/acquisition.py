import numpy as np
from torch.nn.functional import softmax

class AcquisitionFn:
    def __init__(self, model, num_dropouts=10, nb_classes=5):
        self.model = model
        self.aq_fn = self._variation_ratios_acquisition
        self.num_dropouts = num_dropouts
        self.nb_classes = nb_classes
        self.threshold = 0.125

    def is_image_certain(self, image):
        vr = self.aq_fn(image)
        if vr > self.threshold:
            return False
        else:
            return True

    def _predict_with_dropout(self, image):
        self.model.train() # Activate dropout for prediction
        output = self.model(image)
        return softmax(output.cpu()).data.numpy()

    # Measures lack of confidence. Higher values mean less certainty
    def _variation_ratios_acquisition(self, image):
        values = np.zeros((1,self.nb_classes))
        for _ in range(self.num_dropouts):
            prediction = self._predict_with_dropout(image)
            values += prediction
        values /= self.num_dropouts
        vr = 1 - np.max(values)
        return vr
