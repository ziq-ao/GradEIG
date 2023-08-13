from torch import distributions

class NeuralPotentialFunctionForTrueModel:
    """
    Implementation of a potential function for Pyro MCMC which uses the true
    likelihood.
    """

    def __init__(self, sim_model, prior, true_observation):
        """
        :param sim_model: simulation model with 'log_prob' method.
        :param prior: Distribution object with 'log_prob' method.
        :param true_observation: torch.Tensor containing true observation x0.
        """

        self._neural_likelihood = sim_model
        self._prior = prior
        self._true_observation = true_observation

    def __call__(self, inputs_dict):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        :param inputs_dict: dict of parameter values which need evaluation for MCMC.
        :return: torch.Tensor potential ~ -[log q(x0 | theta) + log p(theta)]
        """

        parameters = next(iter(inputs_dict.values()))
        log_likelihood = self._neural_likelihood.log_prob(
            ys=self._true_observation.reshape(1, -1).to("cpu"),
            parameters=parameters.reshape(1, -1),
            design_requires_grad=False
        )

        # If prior is uniform we need to sum across last dimension.
        if isinstance(self._prior, distributions.Uniform):
            potential = -(log_likelihood + self._prior.log_prob(parameters).sum(-1))
        else:
            potential = -(log_likelihood + self._prior.log_prob(parameters))

        return potential
    
class NeuralPotentialFunction:
    """
    Implementation of a potential function for Pyro MCMC which uses a neural density
    estimator to evaluate the likelihood.
    """

    def __init__(self, neural_likelihood, prior, true_observation):
        """
        :param neural_likelihood: Conditional density estimator with 'log_prob' method.
        :param prior: Distribution object with 'log_prob' method.
        :param true_observation: torch.Tensor containing true observation x0.
        """

        self._neural_likelihood = neural_likelihood
        self._prior = prior
        self._true_observation = true_observation

    def __call__(self, inputs_dict):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        :param inputs_dict: dict of parameter values which need evaluation for MCMC.
        :return: torch.Tensor potential ~ -[log q(x0 | theta) + log p(theta)]
        """

        parameters = next(iter(inputs_dict.values()))
        log_likelihood = self._neural_likelihood.log_prob(
            inputs=self._true_observation.reshape(1, -1).to("cpu"),
            context=parameters.reshape(1, -1),
        )

        # If prior is uniform we need to sum across last dimension.
        if isinstance(self._prior, distributions.Uniform):
            potential = -(log_likelihood + self._prior.log_prob(parameters).sum(-1))
        else:
            potential = -(log_likelihood + self._prior.log_prob(parameters))

        return potential