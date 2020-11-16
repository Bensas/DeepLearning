encoder_layers = [35, 25, 15, 5]
latent_layers = 2
decoder_layers = [5, 15, 25, 35]
index = 0
layers = [0] * (len(encoder_layers) + 1 + len(decoder_layers))
for number in encoder_layers:
    layers[index] = number
    index = index + 1
layers[index] = latent_layers
index = index + 1
for number in decoder_layers:
    layers[index] = number
    index = index + 1
print(layers)