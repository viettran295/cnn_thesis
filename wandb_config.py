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
            'values': [15]
        },
        # 'batch_size': {
            # 'distribution': 'int_uniform',
            # 'min': 10,
            # 'max': 50
            # 'values': [10, 50]
        # },
        'dropout': {
            'values': [0.2, 0.3, 0.4]
        },
        'learning_rate': {
            # 'distribution': 'uniform',
            # 'min': 0.001, 
            # 'max': 0.01
            'values': [0.001, 0.005, 0.01]
        },
        'optimizer':{ 
            'values': ['adam', 'sgd', 'RMSprop', 'Adadelta']
        },
        'activation': {
            'values': ['relu', 'elu', 'tanh']
        }
    }
}