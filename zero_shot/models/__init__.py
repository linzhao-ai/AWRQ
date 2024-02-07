# from . import opt
from . import llama

MODEL_REGISTRY = {
    # 'opt': opt.OPT,
    'llama': llama.LLAMA
}


def get_model(model_name):
    if 'opt' in model_name:
        return MODEL_REGISTRY['opt']
    elif 'llama' in model_name:
        return MODEL_REGISTRY['llama']
    return MODEL_REGISTRY[model_name]
