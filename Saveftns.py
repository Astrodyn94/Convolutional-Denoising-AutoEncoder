import torch

def save_encoder(model, epoch , opt):
    model_out_path = opt.checkpoints_dir + '/' + opt.name +'/' + 'encoder_epoch_{}.pth'.format(epoch)
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def save_decoder(model, epoch , opt):
    model_out_path = opt.checkpoints_dir + '/' + opt.name +'/' + 'decoder_epoch_{}.pth'.format(epoch)
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))