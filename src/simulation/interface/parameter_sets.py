import nfpp_sodium_ion.src.cell_parameters.cell_alpha as cell_alpha

def load_default_parameter_set():
    """Loads the default NFPP Sodium-ion parameter set."""
    return cell_alpha.get_parameter_values()
