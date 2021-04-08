# first line: 36
@memory.cache
def get_model_predictions(args_dict, model_path, loader):

    args = Struct(**args_dict)
    print('path : '+model_path)
    model = load_model(args, model_path, loader.dataset.n_classes)
    outs = []
    # gather predictions for all images in the validation set
    for i, (inputs, labels) in enumerate(loader):
        inputs, _ = transform_data((inputs, labels), use_gpu=True)
        outputs = model(inputs)
        out = torch.sigmoid(outputs).data.cpu().numpy()
        outs.append(out)
    outs = np.concatenate(outs, axis=0)
    return outs
