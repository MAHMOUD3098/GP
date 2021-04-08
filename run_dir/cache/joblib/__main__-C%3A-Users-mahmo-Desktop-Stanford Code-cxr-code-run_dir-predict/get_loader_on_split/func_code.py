# first line: 1
import torch

import util
from predict import Struct, memory


@memory.cache
def get_loader_on_split(args_dict, split):

    if "tag" not in args_dict:
        args_dict["tag"] = args_dict["size"]

    args = Struct(**args_dict)

    dataset = util.Dataset(args, split)

    loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=False)
    return loader
