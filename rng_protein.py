import numpy as np

def protein_inputs(yog, mix_p):
    protein = yog["proteins_100g"].to_numpy(dtype=float)
    weights = mix_p["w"]
    mus = mix_p["mu"]
    sigmas = mix_p["sd"]
    return protein, weights, mus, sigmas