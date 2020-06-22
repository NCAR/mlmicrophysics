from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise, Activation, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
import numpy as np
import xarray as xr
import pandas as pd


class DenseNeuralNetwork(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        inputs: Number of input values
        outputs: Number of output values
        activation: Type of activation function
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """
    def __init__(self, hidden_layers=1, hidden_neurons=4, activation="relu",
                 output_activation="linear", optimizer="adam", loss="mse", use_noise=False, noise_sd=0.01,
                 lr=0.001, use_dropout=False, dropout_alpha=0.1, batch_size=128, epochs=2,
                 l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, decay=0, verbose=0,
                 classifier=False):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.loss = loss
        self.lr = lr
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.decay = decay
        self.verbose = verbose
        self.classifier = classifier
        self.y_labels = None
        self.y_labels_val = None
        self.model = None
        self.optimizer_obj = None

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input
        for h in range(self.hidden_layers):
            nn_model = Dense(self.hidden_neurons, activation=self.activation,
                             kernel_regularizer=l2(self.l2_weight), name=f"dense_{h:02d}")(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(nn_model)
        nn_model = Dense(outputs,
                         activation=self.output_activation, name=f"dense_{self.hidden_layers:02d}")(nn_model)
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x, y, xv, yv):
        inputs = x.shape[1]
        if len(y.shape) == 1:
            outputs = 1
        else:
            outputs = y.shape[1]
        if self.classifier:
            outputs = np.unique(y).size
        self.build_neural_network(inputs, outputs)
        self.model.summary()
        if self.classifier:
            self.y_labels = np.unique(y)
            self.y_labels_val = np.unique(yv)
            y_class = np.zeros((y.shape[0], self.y_labels.size), dtype=np.int32)
            y_class_val = np.zeros((yv.shape[0], self.y_labels_val.size), dtype=np.int32)
            for l, label in enumerate(self.y_labels):
                y_class[y == label, l] = 1
            for l, label in enumerate(self.y_labels_val):
                y_class_val[yv == label, l] = 1
            self.model.fit(x, y_class, batch_size=self.batch_size,
                           epochs=self.epochs, verbose=self.verbose,
                           validation_data=(xv, y_class_val))
        else:
            self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs,
                           verbose=self.verbose, validation_data=(xv, yv))
        return self.model.history.history

    def save_fortran_model(self, filename):
        nn_ds = xr.Dataset()
        num_dense = 0
        layer_names = []
        for layer in self.model.layers:
            if "dense" in layer.name:
                layer_names.append(layer.name)
                dense_weights = layer.get_weights()
                nn_ds[layer.name + "_weights"] = ((layer.name + "_in", layer.name + "_out"), dense_weights[0])
                nn_ds[layer.name + "_bias"] = ((layer.name + "_out",), dense_weights[1])
                nn_ds[layer.name + "_weights"].attrs["name"] = layer.name
                nn_ds[layer.name + "_weights"].attrs["activation"] = layer.get_config()["activation"]
                num_dense += 1
        nn_ds["layer_names"] = (("num_layers",), np.array(layer_names))
        nn_ds.attrs["num_layers"] = num_dense
        nn_ds.to_netcdf(filename, encoding={'layer_names':{'dtype': 'S1'}})
        return

    def predict(self, x):
        if self.classifier:
            y_prob = self.model.predict(x, batch_size=self.batch_size)
            y_out = self.y_labels[np.argmax(y_prob, axis=1)].ravel()
        else:
            y_out = self.model.predict(x, batch_size=self.batch_size).ravel()
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob


class DenseGAN(object):
    """
    A conditional generative adversarial network consisting of dense neural networks for the generator
    and discriminator.

    Args:
        hidden_layers: Number of hidden layers in each network
        hidden_neurons: number of neurons in each hidden layer
        activation: Type of nonlinear activation function. Choose from relu, elu, selu, tanh,
        optimizer: Neural network optimization function. Use defaults or pass keras optimizer object.
        loss: Name of loss function being used.
        use_noise: Apply Gaussian noise to hidden layers to generate uncertainty in results
        noise_sd: Standard deviation of Gaussian noise
        use_dropout: Whether or not to include dropout layers
        dropout_alpha: Percent chance of neuron being randomly set to 0. Value should be between 0 and 1.
        batch_norm_output: Whether or not to add batch normalization layer to output of generator
        epochs: Number of epochs to train model
        verbose: If greater than 0, output batch loss values during training
        report_frequency: How often to report the batch loss


    """
    def __init__(self, hidden_layers=2, hidden_neurons=16, activation="relu",
                 optimizer="adam", loss="binary_crossentropy", use_noise=False, noise_sd=0.1, use_dropout=False,
                 dropout_alpha=0.1, batch_norm_output=True, batch_size=1024, epochs=20,
                 verbose=0, report_frequency=50, classifier=False):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.batch_norm_output = batch_norm_output
        self.epochs = epochs
        self.verbose = verbose
        self.report_frequency = report_frequency
        self.classifier = classifier
        self.generator = None
        self.discriminator = None
        self.gen_disc = None
        self.gen_predict_func = None

    def build_generator(self, inputs, outputs):
        gen_input = Input((inputs,))
        gen_model = gen_input
        for h in range(self.hidden_layers):
            gen_model = Dense(self.hidden_neurons)(gen_model)
            gen_model = Activation(self.activation)(gen_model)
            if self.use_noise:
                gen_model = GaussianNoise(self.noise_sd)(gen_model)
            if self.use_dropout:
                gen_model = Dropout(self.dropout_alpha)(gen_model)
        gen_model = Dense(outputs)(gen_model)
        if self.batch_norm_output:
            gen_model = BatchNormalization()(gen_model)
        self.generator = Model(gen_input, gen_model)
        self.generator.compile(optimizer=self.optimizer, loss="mse")

    def build_discriminator(self, inputs, outputs):
        disc_input = Input((outputs,))
        disc_cond_input = Input((inputs,))
        disc_model = Concatenate()([disc_cond_input, disc_input])
        for h in range(self.hidden_layers):
            disc_model = Dense(self.hidden_neurons)(disc_model)
            disc_model = Activation(self.activation)(disc_model)
            if self.use_noise:
                disc_model = GaussianNoise(self.noise_sd)(disc_model)
            if self.use_dropout:
                disc_model = Dropout(self.dropout_alpha)(disc_model)
        disc_model = Dense(1)(disc_model)
        disc_model = Activation("sigmoid")(disc_model)
        self.discriminator = Model([disc_cond_input, disc_input], disc_model)
        self.discriminator.compile(optimizer=self.optimizer, loss=self.loss)

    def stack_gen_disc(self):
        if self.generator is None or self.discriminator is None:
            raise RuntimeError("The generator or discriminator models have not been built yet.")
        self.discriminator.trainable = False
        stacked_model = self.discriminator([self.generator.layers[0].output, self.generator.output])
        self.gen_disc = Model(self.generator.input, stacked_model)
        self.gen_disc.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x, y):
        inputs = x.shape[1]
        outputs = y.shape[1]
        # Build generator model
        self.build_generator(inputs, outputs)
        # Build discriminator model
        self.build_discriminator(inputs, outputs)
        # Stack generator and discriminator models for training the generator
        self.stack_gen_disc()
        self.gen_predict_func = K.function([self.generator.input, K.learning_phase()], [self.generator.output])
        batch_half = int(self.batch_size // 2)
        # Remove examples until the number of training examples is divisible by the batch size
        batch_diff = x.shape[0] % self.batch_size
        if x.shape[0] % self.batch_size != 0:
            x_sub = x[:x.shape[0] - batch_diff]
            y_sub = y[:y.shape[0] - batch_diff]
        else:
            x_sub = x
            y_sub = y
        indices = np.arange(x_sub.shape[0])
        x_gen_batch = np.zeros((self.batch_size, x_sub.shape[1]))
        x_disc_batch = np.zeros((self.batch_size, x_sub.shape[1]))
        y_disc_batch = np.zeros((self.batch_size, y_sub.shape[1]))
        disc_batch_labels = np.zeros(self.batch_size)
        disc_batch_labels[:batch_half] = 1
        gen_batch_labels = np.ones(self.batch_size)
        loss_history = dict(disc_loss=[], gen_loss=[], step=[], epoch=[], batch=[])
        step = 0
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for b, b_index in enumerate(np.arange(self.batch_size, x_sub.shape[0], self.batch_size * 2)):
                x_disc_batch[:] = x_sub[indices[b_index - self.batch_size: b_index]]
                y_disc_batch[:batch_half] = y_sub[indices[b_index - self.batch_size: b_index - batch_half]]
                y_disc_batch[batch_half:] = self.gen_predict_func([x_disc_batch[batch_half:], 1])[0]
                loss_history["disc_loss"].append(self.discriminator.train_on_batch([x_disc_batch, y_disc_batch],
                                                                                   disc_batch_labels))
                x_gen_batch[:] = x_sub[indices[b_index: b_index + self.batch_size]]
                loss_history["gen_loss"].append(self.gen_disc.train_on_batch(x_gen_batch,
                                                                             gen_batch_labels))
                loss_history["epoch"].append(epoch)
                loss_history["batch"].append(b)
                loss_history["step"].append(step)
                if self.verbose > 0 and step % self.report_frequency == 0:
                    print("Epoch {0:03d}, Batch {1:04d}, Step {2:06d}, Disc Loss: {3:0.4f} Gen Loss: {4:0.4f}".format(
                        loss_history["epoch"][-1], loss_history["batch"][-1], loss_history["step"][-1],
                        loss_history["disc_loss"][-1], loss_history["gen_loss"][-1]))
                step += 1
        return pd.DataFrame(loss_history)

    def predict(self, x):
        predictions = self.generator.predict(x).ravel()
        return predictions


