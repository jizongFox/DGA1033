from .loss import CrossEntropyLoss2d, MSE_2D,PartialCrossEntropyLoss2d,negativePartialCrossEntropyLoss2d

LOSS = {'cross_entropy': CrossEntropyLoss2d,
        'mse_2d': MSE_2D,
        'partial_ce':PartialCrossEntropyLoss2d,
        'neg_partial_ce':negativePartialCrossEntropyLoss2d}


def get_loss_fn(name: str, **kwargs):
    try:
        return LOSS.get(name)(**kwargs)
    except Exception as e:
        raise ValueError('name error when inputting the loss name, with %s'%str(e))
