import numpy as np
from src.MetricFunctions import MetricFunction, get_metric_instance, Accuracy


class EarlyStopping():

    """
    Implementation of early stopping.
    
    Attributes
    ----------
    self.metric (MetricFunctioon) : instance of metric used for evaluation
    self.patience (int) : patience of the early stopping
    self.tolerance (float) : tolerance of the early stopping
    self.mode (str) : "min" (if evaluating on error) or "max" (if evaluating on accuracy)
    self._best_metric_value (float) : best metric value in all epochs up to current epoch
    self._n_epochs (int) : current epoch number
    self._n_worsening_epochs (int) : number of consecutive epochs without significant improvement
    self._best_params (list) : weights and biases (for every layer) for model yielding best metric value
    self._best_epoch (int) : epoch on which best metric value was achieved

    Methods
    -------
    __init__ : initializes early stopping object
    initialize : initializes parameters before each training cycle
    on_epoch_end : evaluates performance on validation set at the end of every epoch, eventually saving best parameters

    """

    def __init__ (self,  patience, tolerance, metric):

        """
        Initialize EarlyStopping object.
        
        Parameters
        ----------
        metric (MetricFunction) : instance of metric used for evaluation
        patience (float) : number of consecutive worsening epochs allowed before early stopping
        tolerance (float) : minimum change in the monitored quantity to qualify as improvement
        mode (str, default = "min") : can be both "min" and "max", "max" needed for accuracy (in general, positive metrics)

        """

        self.metric = metric
        self.patience = patience
        self.tolerance = tolerance

        if isinstance(self.metric, Accuracy):
            self.mode = "max"
        else:
            self.mode = "min"
        
        if self.mode == 'min':
            self._best_metric_value = np.infty 
        elif self.mode == 'max':
            self._best_metric_value = -np.infty
        self._n_epochs = 0
        self._n_worsening_epochs = 0

    def on_epoch_end(self, y_true, y_pred, params):

        """
        Evaluates performance on validation set at the end of every epoch, eventually saving best parameters.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values for the validation set
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values for the validation set
        params (list): the current parameters of the model

        Returns (bool): True if training has to stop, False otherwise.

        """

        self._n_epochs += 1

        metric_value = self.metric(y_true, y_pred)

        if (self.mode == "min" and metric_value < self._best_metric_value - self.tolerance) or (self.mode == "max" and metric_value > self._best_metric_value + self.tolerance):
            self._best_metric_value = metric_value
            self._n_worsening_epochs = 0
            self._best_params = params
            self._best_epoch = self._n_epochs
        else:
            self._n_worsening_epochs += 1
            if self._n_worsening_epochs >= self.patience:
                return True
        
        return False