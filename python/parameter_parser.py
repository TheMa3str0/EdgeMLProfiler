def get_inference_params(config_data):
    input_shape = config_data['network']['input_shape']
    no_inference = config_data['network']['inference_params']['no_inferences']
    return [input_shape, no_inference]

def get_training_params(config_data):
    optimizer_choice = config_data['network']['training_params']['optimizer']
    learning_rate = config_data['network']['training_params']['learning_rate']
    loss_function = config_data['network']['training_params']['loss_function']
    batch_size = config_data['network']['training_params']['batch_size']
    epochs = config_data['network']['training_params']['epochs']
    num_samples = config_data['network']['training_params']['num_samples']
    num_classes = config_data['network']['layers'][-1]['io_shape'][1]
    input_shape = config_data['network']['input_shape']
    task = config_data['network']['task']
    return [optimizer_choice, learning_rate, loss_function, batch_size, epochs, num_samples, num_classes, input_shape, task]

def get_warmup_params(config_data):
    no_operations = config_data['network']['warmup_params']['no_operations']
    return [no_operations]