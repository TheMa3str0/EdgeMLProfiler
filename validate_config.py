import json

def load_config_file(config_file_path):
    try:
        with open(config_file_path, 'r') as file:
            config_data = json.load(file)
        return config_data
    except Exception as e:
        print(f"Error loading the config file: {str(e)}")
        return None

def validate_field(config, field_name, data_type, valid_values=None, required=False):
    if required and field_name not in config:
        return f"Missing '{field_name}' in the configuration."

    if field_name in config:
        if data_type == int:
            if not isinstance(config[field_name], int):
                return f"'{field_name}' must be of type {data_type.__name__}."
            elif config[field_name] < 1:
                return f"'{field_name}' must be a positive integer"
        if data_type == str and not isinstance(config[field_name], str):
            return f"'{field_name}' must be of type {data_type.__name__}."

        if valid_values is not None and config[field_name] not in valid_values:
            return f"'{field_name}' must be one of {', '.join(valid_values)}."

    return None

def validate_conv_params(conv_params):
    errors = []

    valid_padding = ['same']
    if 'padding' in conv_params:
        padding_value = conv_params['padding']
        if isinstance(padding_value, int):
            if padding_value < 0:
                errors.append("'padding' value must be a non-negative integer.")
        elif padding_value not in valid_padding:
            errors.append("'padding' must be an integer or 'same'.")

    errors.append(validate_field(conv_params, 'kernel_size', list))
    errors.append(validate_field(conv_params, 'stride', list))
    
    for param_name in ['kernel_size', 'stride']:
        if param_name in conv_params:
            param_value = conv_params[param_name]
            if len(param_value) != 2:
                errors.append(f"'{param_name}' must be a list of 2 elements.")
            for dim in param_value:
                if not isinstance(dim, int) or dim <= 0:
                    errors.append(f"All elements in '{param_name}' must be positive integers.")

    return [error for error in errors if error is not None]

def validate_pool_params(pool_params):
    errors = []

    if 'padding' in pool_params:
        padding_value = pool_params['padding']
        if isinstance(padding_value, int):
            if padding_value < 0:
                errors.append("'padding' value must be a non-negative integer.")
        else:
            errors.append("'padding' must be an integer.")

    errors.append(validate_field(pool_params, 'kernel_size', list))
    errors.append(validate_field(pool_params, 'stride', list))
    
    for param_name in ['kernel_size', 'stride']:
        if param_name in pool_params:
            param_value = pool_params[param_name]
            if len(param_value) != 2:
                errors.append(f"'{param_name}' must be a list of 2 elements.")
            for dim in param_value:
                if not isinstance(dim, int) or dim <= 0:
                    errors.append(f"All elements in '{param_name}' must be positive integers.")

    return [error for error in errors if error is not None]

def validate_dropout_params(dropout_params):
    errors = []
    
    if not isinstance(dropout_params['p'], (float, int)) or dropout_params['p'] < 0:
        errors.append("'p (probability)' must be a positive number.")
        
    if not isinstance(dropout_params['inplace'], bool):
        errors.append("'inplace' must be boolean.")
        
    return [error for error in errors if error is not None]

def validate_residual_params(residual_params):
    errors = []
    
    if not isinstance(residual_params['in_channels'], int) or residual_params['in_channels'] < 0:
        errors.append("'in_channels' must be a positive number.")
        
    if not isinstance(residual_params['out_channels'], int) or residual_params['out_channels'] < 0:
        errors.append("'out_channels' must be a positive number.")
        
    if not isinstance(residual_params['stride'], int) or residual_params['stride'] < 0:
        errors.append("'stride' must be a positive number.")
        
    return [error for error in errors if error is not None]

def check_network_config(network_config):
    errors = []

    valid_devices = ['cpu', 'gpu']
    errors.append(validate_field(network_config, 'device', str, valid_values=valid_devices))
    errors.append(validate_field(network_config, 'input_shape', list, required=True))

    input_shape = network_config.get('input_shape')
    if input_shape:
        if len(input_shape) < 1:
            errors.append("'input_shape' must not be empty.")
        for dim in input_shape:
            if not isinstance(dim, int) or dim <= 0:
                errors.append("All elements in 'input_shape' must be positive integers.")
    
    valid_modes = ['inference', 'training']
    errors.append(validate_field(network_config, 'mode', str, valid_values=valid_modes, required=True))
    
    valid_tasks = ['classification', 'regression']
    errors.append(validate_field(network_config, 'task', str, valid_values=valid_tasks, required=True))

    if network_config['mode'] == 'inference':
        inference_params = network_config.get('inference_params')
        if not inference_params:
            errors.append("Missing 'inference_params' in 'inference' mode.")
        else:
            errors.append(validate_field(inference_params, 'no_inferences', int, required=True))

    if network_config['mode'] == 'training':
        training_params = network_config.get('training_params')
        if not training_params:
            errors.append("Missing 'training_params' in 'training' mode.")
        else:
            valid_optimizers = ['adam', 'sgd', 'rmsprop']
            errors.append(validate_field(training_params, 'optimizer', str, valid_values=valid_optimizers, required=True))
            errors.append(validate_field(training_params, 'learning_rate', float, required=True))
            valid_losses = ['categorical_crossentropy', 'mse']
            errors.append(validate_field(training_params, 'loss_function', str, valid_values=valid_losses, required=True))
            errors.append(validate_field(training_params, 'batch_size', int, required=True))
            errors.append(validate_field(training_params, 'epochs', int, required=True))
            errors.append(validate_field(training_params, 'num_samples', int, required=True))
            
    warmup_params = network_config.get('warmup_params')
    if not warmup_params:
        errors.append("Missing 'warmup_params' in the config file.")
    else:
        if 'no_operations' not in warmup_params:
            errors.append("Missing 'no_operations' in the 'warmup_params.")
        if not isinstance(warmup_params['no_operations'], int):
            errors.append("'no_operations' must be of type int")
        if warmup_params['no_operations'] < 0:
            errors.append("'no_operations' value must be a non-negative integer")

    # Add more checks for specific keys in the 'network' section

    return [error for error in errors if error is not None]

def check_layers_config(layers):
    errors = []

    for layer in layers:
        valid_layer_types = ['conv2d', 'dense', 'maxpool2d', 'flatten', 'batchnorm2d', 'averagepool2d', 'dropout', 'residual_block']
        errors.append(validate_field(layer, 'type', str, valid_values=valid_layer_types, required=True))
        
        valid_activation_functions = ['relu', 'softmax', 'tanh']
        errors.append(validate_field(layer, 'activation_function', str, valid_values=valid_activation_functions))
        errors.append(validate_field(layer, 'io_shape', list))

        io_shape = layer.get('io_shape')
        if io_shape:
            if len(io_shape) != 2:
                errors.append("'io_shape' must be a list of 2 elements.")
            for dim in io_shape:
                if not isinstance(dim, int) or dim <= 0:
                    errors.append("All elements in 'io_shape' must be positive integers.")
            
        if layer['type'] == 'conv2d':
            if 'conv_params' in layer:
                conv_params_errors = validate_conv_params(layer['conv_params'])
                errors.extend(conv_params_errors)
            else:
                errors.append("Missing 'conv_params' in a 'conv2d' layer.")
                
        if layer['type'] == 'maxpool2d':
            if 'maxpool_params' in layer:
                maxpool_params_errors = validate_pool_params(layer['maxpool_params'])
                errors.extend(maxpool_params_errors)
            else:
                errors.append("Missing 'maxpool_params' in a 'maxpool2d' layer.")
                
        if layer['type'] == 'batchnorm2d':
            if 'batchnorm_params' in layer:
                if len(layer['batchnorm_params']['num_features']) != 1:
                    errors.append("'num_features' must be a list of 1 element.")
                for dim in layer['batchnorm_params']['num_features']:
                    if not isinstance(dim, int) or dim <= 0:
                        errors.append("All elements in 'num_features' must be positive integers.")
            else:
                errors.append("Missing 'batchnorm_params' in a 'batchnorm2d' layer.")
                
        if layer['type'] == 'dropout':
            if 'dropout_params' in layer:
                dropout_errors = validate_dropout_params(layer['dropout_params'])
                errors.extend(dropout_errors)
            else:
                errors.append("Missing 'p (probability)' in a 'dropout' layer.")
                
        if layer['type'] == 'averagepool2d':
            if 'averagepool_params' in layer:
                averagepool_params_errors = validate_pool_params(layer['averagepool_params'])
                errors.extend(averagepool_params_errors)
            else:
                errors.append("Missing 'averagepool_params' in a 'averagepool2d' layer.")
                
        if layer['type'] == 'residual_block':
            if 'residual_params' in layer:
                residual_params_errors = validate_residual_params(layer['residual_params'])
                errors.extend(residual_params_errors)
            else:
                errors.append("Missing 'residual_params' in a 'residual_block'.")

        # Add more checks for specific keys in different types of layers

    return [error for error in errors if error is not None]

def validate_config(config_file_path):
    config_data = load_config_file(config_file_path)

    if config_data is not None:
        network_errors = check_network_config(config_data['network'])
        layers = config_data['network']['layers']
        layers_errors = check_layers_config(layers)

        all_errors = network_errors + layers_errors

        if all_errors:
            print("Configuration Errors:")
            for error in all_errors:
                print(error)
            return False
        else:
            print("Config file is valid.")
            return True
    else:
        print("Error loading config file.")
        return False
