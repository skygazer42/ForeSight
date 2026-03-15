import torch

class QuantileLossCalculator():
    """Computes the combined quantile loss for prespecified quantiles.
    Attributes:
      quantiles: Quantiles to compute losses
    """

    def __init__(self, quantiles, output_size):
        """Initializes computer with quantiles for loss calculations.
            Args:
            quantiles: Quantiles to use for computations.
        """
        self.quantiles = quantiles
        self.output_size = output_size
        
    # Loss functions.
    def quantile_loss(self, y, y_pred, quantile):
        """ Computes quantile loss for pytorch.
            Standard quantile loss as defined in the "Training Procedure" section of
            the main TFT paper
            Args:
            y: Targets
            y_pred: Predictions
            quantile: Quantile to use for loss calculations (between 0 & 1)
            Returns:
            Tensor for quantile loss.
        """

        # Checks quantile
        if quantile < 0 or quantile > 1:
            raise ValueError(
                'Illegal quantile value={}! Values should be between 0 and 1.'.format(quantile))

        prediction_underflow = y - y_pred
        q_loss = quantile * torch.max(prediction_underflow, torch.zeros_like(prediction_underflow)) + \
                (1. - quantile) * torch.max(-prediction_underflow, torch.zeros_like(prediction_underflow))
        return q_loss.unsqueeze(1)

    def apply(self, b, a):
        """Returns quantile loss for specified quantiles.
            Args:
            a: Targets
            b: Predictions
        """
        quantiles_used = set(self.quantiles)

        loss = []
        for i, quantile in enumerate(self.quantiles):
            if quantile in quantiles_used:
                loss.append(self.quantile_loss(a[Ellipsis, i],
                                               b[Ellipsis, i], 
                                               quantile))

        loss_computed = torch.mean(torch.sum(torch.cat(loss, axis = 1), axis = 1))
        
        return loss_computed

class NormalizedQuantileLossCalculator():
    """Computes the combined quantile loss for prespecified quantiles.
    Attributes:
      quantiles: Quantiles to compute losses
    """

    def __init__(self, quantiles, output_size):
        """Initializes computer with quantiles for loss calculations.
            Args:
            quantiles: Quantiles to use for computations.
        """
        self.quantiles = quantiles
        self.output_size = output_size
        
    # Loss functions.
    def apply(self, y, y_pred, quantile):
        """ Computes quantile loss for pytorch.
            Standard quantile loss as defined in the "Training Procedure" section of
            the main TFT paper
            Args:
            y: Targets
            y_pred: Predictions
            quantile: Quantile to use for loss calculations (between 0 & 1)
            Returns:
            Tensor for quantile loss.
        """

        # Checks quantile
        if quantile < 0 or quantile > 1:
            raise ValueError(
                'Illegal quantile value={}! Values should be between 0 and 1.'.format(quantile))

        prediction_underflow = y - y_pred
        weighted_errors = quantile * torch.max(prediction_underflow, torch.zeros_like(prediction_underflow)) + \
                (1. - quantile) * torch.max(-prediction_underflow, torch.zeros_like(prediction_underflow))
        
        quantile_loss = torch.mean(weighted_errors)
        normaliser = torch.mean(torch.abs(quantile_loss))
        return 2 * quantile_loss / normaliser
