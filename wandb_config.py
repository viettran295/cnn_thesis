config_default = {
    'activation': 'elu',
    'epoch': 50,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'loss': 'mse',
    'optimizer': 'adam'
}

sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters':{
        'epochs': {
            'values': [10]
        },
        'batch_size': {
            'distribution': 'int_uniform',
            'min': 120,
            'max': 240
            # 'values': [10, 20, 40, 60, 80, 100, 120]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.7
            # 'values': [0.2, 0.4, 0.6, 0.8]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.00001, 
            'max': 0.001
            # 'values': [0.001, 0.0025, 0.005, 0.0075, 0.01]
        },
        'optimizer':{ 
            'values': ['Adam']
        },
        'activation': {
            'values': ['tanh']
        }
    }
}