from keras.layers import Input, Dense, Dropout, GaussianNoise, Activation, Concatenate, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
import keras.backend as K
import numpy as np
from scipy.stats import norm, randint, uniform, expon
from sklearn.model_selection import ParameterSampler


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
    def __init__(self, hidden_layers=1, hidden_neurons=4, inputs=1, outputs=1, activation="relu",
                 output_activation="linear", optimizer="adam", loss="mse", use_noise=False, noise_sd=0.01,
                 lr=0.001, use_dropout=False, dropout_alpha=0.1, batch_size=128, epochs=2,
                 l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, decay=0, verbose=0):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.inputs = inputs
        self.outputs = outputs
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
        nn_input = Input(shape=(self.inputs,))
        nn_model = nn_input
        for h in range(self.hidden_layers):
            nn_model = Dense(self.hidden_neurons, kernel_regularizer=l2(self.l2_weight))(nn_model)
            nn_model = Activation(self.activation)(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha)(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd)(nn_model)
        nn_model_out = Dense(int(self.outputs))(nn_model)
        #nn_model_out = []
        #for i in range(self.outputs):
        #    nn_model_out.append(Dense(self.hidden_neurons, kernel_regularizer=l2(self.l2_weight))(nn_model))
        #    nn_model_out[-1] = Activation(self.activation)(nn_model_out[-1])
        #    nn_model_out[-1] = Dense(1, activation=output_activation)(nn_model_out[-1])
        self.model = Model(nn_input, nn_model_out)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        print(self.model.summary())

    def fit(self, x, y):
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        return

    def predict(self, x):
        return np.array(self.model.predict(x, batch_size=self.batch_size))


class DenseGAN(object):
    """
    A conditional generative adversarial network consisting of dense neural networks for the generator
    and discriminator.

    Args:
        hidden_layers: Number of hidden layers in each network
        hidden_neurons: number of neurons in each hidden layer


    """
    def __init__(self, hidden_layers=2, hidden_neurons=16, inputs=1, random_inputs=1, outputs=1, activation="relu",
                 optimizer="adam", loss="binary_crossentropy", use_noise=False, noise_sd=0.1, use_dropout=False,
                 dropout_alpha=0.1, l2_weight=0.01, batch_norm_output=True, random_dist=norm(), batch_size=1024, epochs=20,
                 verbose=0, report_frequency=50):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.inputs = inputs
        self.random_inputs = random_inputs
        self.outputs = outputs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.l2_weight = l2_weight
        self.dropout_alpha = dropout_alpha
        self.batch_norm_output = batch_norm_output
        self.epochs = epochs
        self.random_dist = random_dist
        self.verbose = verbose
        self.report_frequency = report_frequency
        self.generator = None
        self.discriminator = None
        self.gen_disc = None
        # Build generator model
        self.build_generator()
        # Build discriminator model
        self.build_discriminator()
        # Stack generator and discriminator models for training the generator
        self.stack_gen_disc()
        self.gen_predict_func = K.function(self.generator.input + [K.learning_phase()], [self.generator.output])

    def build_generator(self):
        gen_input = Input((self.inputs,))
        random_input = Input((self.random_inputs,))
        gen_model = Concatenate()([gen_input, random_input])
        for h in range(self.hidden_layers):
            if self.use_noise:
                gen_model = GaussianNoise(self.noise_sd)(gen_model)
            gen_model = Dense(self.hidden_neurons, kernel_regularizer=l2(self.l2_weight))(gen_model)
            gen_model = Activation(self.activation)(gen_model)
            if self.use_dropout:
                gen_model = Dropout(self.dropout_alpha)(gen_model)
        gen_model = Dense(self.outputs)(gen_model)
        if self.batch_norm_output:
            gen_model = BatchNormalization()(gen_model)
        self.generator = Model([gen_input, random_input], gen_model)
        self.generator.compile(optimizer=self.optimizer, loss="mse")

    def build_discriminator(self):
        disc_input = Input((self.outputs,))
        disc_cond_input = Input((self.inputs,))
        disc_model = Concatenate()([disc_cond_input, disc_input])
        for h in range(self.hidden_layers):
            if self.use_noise:
                disc_model = GaussianNoise(self.noise_sd)(disc_model)
            disc_model = Dense(self.hidden_neurons, kernel_regularizer=l2(self.l2_weight))(disc_model)
            disc_model = Activation(self.activation)(disc_model)
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
        stacked_model = self.discriminator(self.generator.output)
        self.gen_disc = Model(self.generator.input, stacked_model)
        self.gen_disc.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x, y):
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
        gen_random_batch = np.zeros((self.batch_size, self.random_inputs))
        disc_random_batch = np.zeros((batch_half, self.random_inputs))
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
                gen_random_batch[:] = self.random_dist.rvs(size=(self.batch_size, self.random_inputs))
                disc_random_batch[:] = self.random_dist.rvs(size=(batch_half, self.random_inputs))
                y_disc_batch[:batch_half] = y_sub[indices[b_index - self.batch_size: b_index - batch_half]]
                y_disc_batch[batch_half:] = self.gen_predict_func([x_sub[b_index - batch_half: b_index],
                                                                   disc_random_batch, 1])[0]
                x_disc_batch[:] = x_sub[b_index - self.batch_size: b_index]
                loss_history["disc_loss"].append(self.discriminator.train_on_batch([x_disc_batch, y_disc_batch],
                                                                                   disc_batch_labels))
                x_gen_batch[:] = x_sub[b_index: b_index + self.batch_size]
                loss_history["gen_loss"].append(self.gen_disc.train_on_batch([x_gen_batch, gen_random_batch],
                                                                             gen_batch_labels))
                loss_history["epoch"].append(epoch)
                loss_history["batch"].append(b)
                loss_history["step"].append(step)
                if self.verbose > 0 and step % self.report_frequency == 0:
                    print("Epoch {0:03d}, Batch {1:04d}, Step {2:06d}, Disc Loss: {3:0.4f} Gen Loss: {4:0.4f}".format(
                        loss_history["epoch"][-1], loss_history["batch"][-1], loss_history["step"][-1],
                        loss_history["disc_loss"][-1], loss_history["gen_loss"][-1]))
                step += 1

    def predict(self, x):
        if not isinstance(x, list):
            predictions = self.generator.predict([x, np.random.normal(size=(x.shape[0], self.random_inputs))])
        else:
            predictions = self.generator.predict(x)
        return predictions


def parse_model_config_params(model_params, num_settings, random_state):
    param_distributions = dict()
    dist_types = dict(randint=randint, expon=expon, uniform=uniform)
    for param, param_value in model_params.items():
        if param_value[0] in ["randint", "expon", "uniform"]:
            param_distributions[param] = dist_types[param_value[0]](*param_value[1:])
        else:
            param_distributions[param] = param_value
    return ParameterSampler(param_distributions, n_iter=num_settings, random_state=random_state)

