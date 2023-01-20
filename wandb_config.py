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
            'values': [8]
        },
        'batch_size': {
            'distribution': 'int_uniform',
            'min': 10,
            'max': 200
            # 'values': [10, 20, 40, 60, 80, 100, 120, 140]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.7
            # 'values': [0, 0.2, 0.4, 0.6]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0001, 
            'max': 0.01
            # 'values': [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]
        },
        'optimizer':{ 
            'values': ['Adam']
        },
        'activation': {
            'values': ['elu']
        }
    }
}