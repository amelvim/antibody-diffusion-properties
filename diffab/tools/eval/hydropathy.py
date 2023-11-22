import numpy as np
from diffab.tools.eval.base import EvalTask
from diffab.tools.eval.similarity import entity_to_seq, extract_reslist
from diffab.utils.protein.constants import ressymb_to_resindex


# Dictionary to convert amino acids to their hydropathy index
hydropathy_dict = {
    "C": 2.5,  "D": -3.5, "S": -0.8, "Q": -3.5, "K": -3.9,
    "I": 4.5,  "P": -1.6, "T": -0.7, "F": 2.8,  "N": -3.5,
    "G": -0.4, "H": -3.2, "L": 3.8,  "R": -4.5, "W": -0.9,
    "A": 1.8,  "V": 4.2,  "E": -3.5, "Y": -1.3, "M": 1.9,
    "X": 0,    "-": 0
}


# Generate a vector with hydropathy values for amino acids sorted as in `ressymb_to_resindex`
hydropathy_vector = np.zeros((len(ressymb_to_resindex)))
for aa_res, index in ressymb_to_resindex.items():
    value = hydropathy_dict[aa_res]
    hydropathy_vector[index] = value


# Convert hydropathy values into probabilities (remove amino acid `X` value from vector)
hydropathy_prob = -hydropathy_vector[:-1] + hydropathy_vector[:-1].max()
hydropathy_prob /= hydropathy_prob.sum()


def get_hydropathy(aa_seq):
    return np.array([hydropathy_dict[a] for a in aa_seq])


def eval_hydropathy(task: EvalTask):
    model_gen = task.get_gen_biopython_model()
    reslist_gen = extract_reslist(model_gen, task.residue_first, task.residue_last)
    seq_gen, _ = entity_to_seq(reslist_gen)
    hydro_avg = get_hydropathy(seq_gen).mean()

    task.scores.update({'hydro': hydro_avg})
    return task
