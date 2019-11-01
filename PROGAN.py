import torch as th
import torchvision as tv
import pro_gan_pytorch.PRO_GAN as pg

#th.cuda.set_device(1)
#device=th.device('cuda:2')
#device = th.device("cuda" if th.cuda.is_available() else "cpu")
device=th.cuda.set_device(1)
train_path = "/media/reserve_storage/student_data/BE21_Soham/data/train/"

val_path = "/media/reserve_storage/student_data/BE21_Soham/data/val/"

transform=tv.transforms.Compose([tv.transforms.Resize([1024,1024]),tv.transforms.ToTensor()])

trainset=tv.datasets.ImageFolder(root=train_path,transform=transform)

valset=tv.datasets.ImageFolder(root=val_path,transform=transform)

depth=4

num_epochs = [20, 20, 20, 20]
fade_ins = [50, 50, 50, 50]
batch_sizes = [8,8,8,8]
latent_size = 128

pro_gan = pg.ConditionalProGAN(num_classes=2, depth=depth, 
                                   latent_size=latent_size, device=device)
                                   
pro_gan.train(
        dataset=trainset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes
    )
