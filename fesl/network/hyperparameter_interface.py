"""Interface function to get the correct type of hyperparameter."""
from .hyperparameter_oat import HyperparameterOAT
from .hyperparameter_optuna import HyperparameterOptuna


def HyperparameterInterface(hotype, opttype="float", name="", low=0, high=0,
                            choices=[]):
    """
    Get the correct type of hyperparameter based on your hotype.

    Parameters
    ----------
    hotype : string
        Type of hyperparameter. Currently supported:

            - optuna: Optuna style hyperparameter, for HyperOptOptuna.
            - oat: Hyperparameter for HyperOptOAT.

        opttype : string
            Datatype of the hyperparameter. Follows optunas naming convetions.
            In principle supported are:

                - float
                - int
                - categorical (list)

            Float and int are not available for OA based approaches at the
            moment.

        name : string
            Name of the hyperparameter. Please note that these names always
            have to be distinct; if you e.g. want to investigate multiple
            layer sizes use e.g. ff_neurons_layer_001, ff_neurons_layer_002, etc. as names.

        low : float or int
            Lower bound for numerical parameter.

        high : float or int
            Higher bound for numerical parameter.

        choices :
            List of possible choices (for categorical parameter).

    Returns
    -------
    hyperparameter : HyperparameterOptuna or HyperparameterOAT
        Hyperparameter in desired format.
    """
    hparam = None
    if hotype == "optuna":
        hparam = HyperparameterOptuna(opttype=opttype, name=name, low=low, high=high, choices=choices)
    elif hotype == "oat":
        hparam = HyperparameterOAT(opttype=opttype, name=name, choices=choices)

    if hparam is not None:
        return hparam
    else:
        raise Exception("Unknown hyperparameter type..")
