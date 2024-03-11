class History:
    def __init__(self, loss, val_loss, rmse_v, val_rmse_v, rmse_a, val_rmse_a):
        self.loss = loss
        self.val_loss = val_loss
        self.rmse_valence = rmse_v
        self.val_rmse_valence = val_rmse_v
        self.rmse_arousal = rmse_a
        self.val_rmse_arousal = val_rmse_a

