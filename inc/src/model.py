# Input and output shapes
inputShape = (None, None, ModelConfig.L_FRAME // 2 + 1)
outputMusicShape = (None, None, ModelConfig.L_FRAME // 2 + 1)
outputVoiceShape = (None, None, ModelConfig.L_FRAME // 2 + 1)
# TF graph input
self.input = tf.placeholder(tf.float32, shape=inputShape, name='mix')
# TF graph output
self.music = tf.placeholder(tf.float32, shape=outputMusicShape, name='music')
self.voice = tf.placeholder(tf.float32, shape=outputVoiceShape, name='voice')
# Network
self.hiddenSize = hiddenSize  # Size of hidden layers
self.layerCount = nRnnLayer  # Count of hidden layers
self.network = tf.make_template('network', self.netGen)
