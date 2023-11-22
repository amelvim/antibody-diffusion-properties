from copy import deepcopy
import torch
from .models.predictor import DDGPredictor
from .utils.data import load_wt_mut_pdb_pair
from .utils.misc import recursive_to
from .utils.protein import ATOM_CA


MODELNAME = "./diffab/tools/ddg/data/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = torch.load(MODELNAME, map_location=DEVICE)
MODEL = DDGPredictor(CKPT["config"].model)
MODEL.load_state_dict(CKPT["model"])
MODEL.to(DEVICE)
MODEL.eval()


def predict_ddg(wt_pdb, mut_pdb):
    batch = load_wt_mut_pdb_pair(wt_pdb, mut_pdb)
    batch = recursive_to(batch, DEVICE)

    with torch.no_grad():
        pred = MODEL(batch["wt"], batch["mut"]).item()

    return pred


def convert_batch_wt(batch_ref):
    batch_wt = dict()

    batch_wt["chain_id"] = batch_ref["chain_id"]
    batch_wt["chain_seq"] = batch_ref["chain_nb"]
    batch_wt["resseq"] = batch_ref["resseq"]
    batch_wt["seq"] = batch_ref["res_nb"]
    batch_wt["aa"] = batch_ref["aa"]

    batch_wt["pos14"] = batch_ref["pos_heavyatom"][:, :, :-1, :] # 15 to 14 atoms
    mask_heavyatom = batch_ref["mask_heavyatom"][:, :, :-1] # 15 to 14 atoms
    batch_wt["pos14_mask"] = mask_heavyatom.unsqueeze(-1).repeat(1, 1, 1, 3) # repeat last dimension

    mutation_mask = batch_ref["generate_flag"]
    mask = batch_ref["mask"]

    return batch_wt, mutation_mask, mask


def create_batch_mut(batch_wt, aa_mut, pos_mut, mutation_mask):
    batch_mut = deepcopy(batch_wt)

    # Replace amino acid sequence
    aa_wt = batch_wt["aa"]
    batch_mut["aa"] = torch.where(mutation_mask, aa_mut, aa_wt)

    # Replace Ca position (zero the rest)
    pos14_mut = torch.zeros_like(batch_wt["pos14"])
    pos14_mask_mut = torch.zeros_like(batch_wt["pos14_mask"])
    pos14_mut[:, :, ATOM_CA] = pos_mut
    pos14_mask_mut[:, :, ATOM_CA] = True
    batch_mut["pos14"] = pos14_mut
    batch_mut["pos14_mask"] = pos14_mask_mut

    return batch_mut


def predict_ddg_batch(batch_ref, aa_gen, pos_gen, aa_ref=None, pos_ref=None):
    batch_wt, mutation_mask, mask = convert_batch_wt(batch_ref)
    if (aa_ref is not None) and (pos_ref is not None): # modify reference if provided
        batch_wt = create_batch_mut(batch_wt, aa_ref, pos_ref, mutation_mask)
    batch_mut = create_batch_mut(batch_wt, aa_gen, pos_gen, mutation_mask)

    batch = {"wt": batch_wt, "mut": batch_mut,
             "mutation_mask": mutation_mask, "mask": mask} # no kNN, no Padding

    with torch.no_grad():
        pred = MODEL(batch["wt"], batch["mut"])

    return pred
