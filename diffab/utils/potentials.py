import sys
import torch

from diffab.utils.protein.constants import ressymb_to_resindex

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# TEMPLATE CLASS
class Potential:
    def get_gradients(seq):
        sys.exit("ERROR POTENTIAL HAS NOT BEEN IMPLEMENTED")


class HydropathyIndex(Potential):
    """ Calculate loss with respect to soft_seq of the sequence hydropathy index
        (Kyte and Doolittle, 1986).
    """
    def __init__(self, target_score, potential_scale=1, loss_type="simple"):

        self.target_score = target_score
        self.potential_scale = potential_scale
        self.loss_type = loss_type
        print(f'USING {self.loss_type} LOSS TYPE...')

        # AA conversion
        # self.conversion = list("ARNDCQEGHILKMFPSTWYV")
        self.conversion = list(ressymb_to_resindex.keys())[:20]

        # Dictionary to convert amino acids to their hydropathy index
        self.hydropathy_dict = {
            'C': 2.5, 'D': -3.5, 'S': -0.8, 'Q': -3.5, 'K': -3.9,
            'I': 4.5, 'P': -1.6, 'T': -0.7, 'F': 2.8, 'N': -3.5,
            'G': -0.4, 'H': -3.2, 'L': 3.8, 'R': -4.5, 'W': -0.9,
            'A': 1.8, 'V':4.2, 'E': -3.5, 'Y': -1.3, 'M': 1.9,
            'X': 0, '-': 0
        }
        self.hydropathy_list = [self.hydropathy_dict[a] for a in self.conversion]

        print(f'GUIDING SEQUENCES TO HAVE TARGET HYDROPATHY SCORE OF: {self.target_score}')


    @torch.enable_grad()
    def get_gradients(self, seq, mask_generate=None):
        """
        Calculate gradients with respect to hydropathy index of input seq.
        Uses a MSE loss.
        Args:
            seq:            input sequence, (N, L, 20).
            mask_generate:  (N, L).
        Returns:
            gradients:      gradients of soft_seq with respect to loss on target score, (N, L, 20).
        """

        # Get matrix based on length of seq
        hydropathy_matrix  = torch.tensor(self.hydropathy_list)[None, None].repeat(seq.shape[0], seq.shape[1], 1)
        hydropathy_matrix = hydropathy_matrix.requires_grad_(requires_grad=True).to(DEVICE)

        # Get softmax of seq
        soft_seq = torch.softmax(seq, dim=-1).requires_grad_(requires_grad=True)

        # Calculate simple MSE loss on hydropathy_score
        if self.loss_type == 'simple':
            hydropathy_score = torch.mean(
                torch.sum(soft_seq * hydropathy_matrix, dim=-1), dim=-1
            )
            loss = torch.mean((hydropathy_score - self.target_score)**2)**0.5 # RMSE
            # Take backward step
            loss.backward()
            # Get gradients from soft_seq
            gradients = soft_seq.grad

        # Calculate MSE loss on hydropathy_score
        elif self.loss_type == 'complex':
            loss = torch.mean(
                (torch.sum(soft_seq * hydropathy_matrix, dim=-1) - self.target_score)**2
            )
            # Take backward step
            loss.backward()
            # Get gradients from soft_seq
            gradients = soft_seq.grad

        gradients *= self.potential_scale
        if mask_generate is not None:
            gradients = torch.where(mask_generate[..., None].expand(gradients.size()), gradients, 1)

        return -gradients
