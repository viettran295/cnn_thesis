config_defaults = {
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
            'min': 5,
            'max': 15
        },
        'batch_size': {
            'min': 10,
            'max': 100
        },
        # 'dropout': {
        #     'values': [0.2, 0.3]
        # },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0001, 
            'max': 0.1
        },
        'optimizer':{ 
            'values': ['adam', 'sgd', 'RMSprop', 'Adadelta']
        },
        'activation': {
            'values': ['relu', 'elu', 'tanh']
        }
    }
}