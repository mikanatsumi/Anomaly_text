
from .ops import createConvFunc

nets = {
   
    
   
    'rrrv4': {
        'layer0':  'rd',
        'layer1':  'rd',
        'layer2':  'rd',
        'layer3':  'cv',
        'layer4':  'rd',
        'layer5':  'rd',
        'layer6':  'rd',
        'layer7':  'cv',
        'layer8':  'rd',
        'layer9':  'rd',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'cv',
     
       
    'carv4': {
        'layer0':  'cd',
        'layer1':  'ad',
        'layer2':  'rd',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'ad',
        'layer6':  'rd',
        'layer7':  'cv',
        'layer8':  'cd',
        'layer9':  'ad',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'ad',
        'layer14': 'rd',
        'layer15': 'cv',
        },
    }


def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(createConvFunc(op))

    return pdcs

def config_model_converted(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(op)

    return pdcs

