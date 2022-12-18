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
            'values': [5]
        },
        'batch_size': {
            # 'distribution': 'int_uniform',
            # 'min': 10,
            # 'max': 50
            'values': [30, 60, 90, 120, 150]
        },
        'dropout': {
            # 'distribution': 'uniform',
            # 'min': 0.1,
            # 'max': 0.7
            'values': [0.2, 0.4, 0.6]
        },
        'learning_rate': {
            # 'distribution': 'uniform',
            # 'min': 0.001, 
            # 'max': 0.01
            'values': [0.0001, 0.0005, 0.001, 0.005, 0.01]
        },
        'optimizer':{ 
            'values': ['Adam', 'SGD', 'RMSProp']
            # 'values': ['adam']
        },
        'activation': {
            'values': ['relu', 'tanh', 'elu'],
            # 'values': ['tanh']
        }
    }
}