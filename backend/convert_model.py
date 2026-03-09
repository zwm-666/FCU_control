import os
import sys

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure the original trainer can be imported
sys.path.insert(0, r'D:\python\pythonProject3\learn\我的模型1')

import tensorflow as tf
from ceo_qaadam_emstgat_trainer import EnhancedMSTGAT, ModelConfig

print('TensorFlow version:', tf.__version__)

# Define the config using the best params found
config = ModelConfig(
    input_shape=(7,),
    num_nodes=7,
    num_classes=4,
    hidden_units=192,
    attention_heads=16,
    dropout_rate=0.20559374642540468,
    max_sequence_length=15,
    knn_top_k=6,
    dilation_rate=4,
    use_batch_norm=False,
    embedding_dim=160
)

print('Building model...')
model = EnhancedMSTGAT(config)
model.build(input_shape=(None, 7))
print('Model built! Total params:', model.count_params())

keras_path = r'D:\python\pythonProject3\learn\我的模型1\results\ceo_qaadam_emstgat.keras'
print(f'Loading original model from: {keras_path}')

# Load the weights from the .keras file
model.load_weights(keras_path)
print('Weights loaded successfully.')

out_path = r'c:\Users\86191\Downloads\h2-fcu-modern-dashboard\backend\models\emstgat_weights.weights.h5'
print(f'Saving weights to: {out_path}')
model.save_weights(out_path)
print('Done!')
