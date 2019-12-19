import numpy as np

class AcquisitionFn:
    @staticmethod
    def _select_aq_fn(name):
        if name == 'max_entropy':
            return self._max_entropy_acquisition
        elif name == 'mean_std':
            return self._mean_std_acquisition
        else:
            raise Exception(f"Invalid acquisition function name: {name}")

    def __init__(self, model, fname='max_entropy', num_dropouts=10, nb_classes=5):
        self.model = model
        self.aq_fn = self._select_aq_fn(fname)
        self.num_dropouts = num_dropouts
        self.nb_classes = nb_classes

    def get_certainty(self, image):
        return self.aq_fn(image)

    def _predict_with_dropout(self, image):
        output = self.model(image)
        return F.softmax(output.cpu()).data.numpy()

    def _max_entropy_acquisition(self, image):
        print("MAX ENTROPY FUNCTION")
        running_score = np.zeros((1, nb_classes))
        self.model.train() # Need dropout during eval predictions

        for _ in range(self.num_dropouts):
            prediction = self._predict_with_dropout(image)
            running_score += prediction

        avg_pi = np.divide(running_score, self.num_dropouts)
        log_avg_pi = np.log2(avg_pi)
        entropy = -np.multiply(avg_pi, log_avg_pi)
        entropy = np.sum(entropy, axis=1)
        return entropy.flatten()

    def _mean_std_acquisition(self, image):
        print("MEAN STD ACQUISITION FUNCTION")
        running_score = np.zeros((1,1))
        self.model.train()

        for _ in range(self.num_dropouts):
            prediction = self._predict_with_dropout(image)
            running_score = np.append(running_score, prediction, axis=1)

        std_devs = np.zeros((1,self.nb_classes))
        sigma = np.zeros(1)
        for t in range(1):
            for r in range(self.nb_classes):
                L = np.array([0])
                for k in range(r+1, running_score.shape[1], 10):
                    L = np.append(L, running_score[t,k])
                L_std = np.std(L[1:])
                std_devs[t,r] = L_std
            E = std_devs[t,:]
            sigma[t] = sum(E)/self.nb_classes

        return sigma.flatten()
