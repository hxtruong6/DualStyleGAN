import os
import sys

CURR_PATH = os.path.dirname(__file__)
print("[StyleTransferService] Dirname:", CURR_PATH)
sys.path.append(CURR_PATH)


from PIL import Image
from argparse import Namespace
from datetime import datetime
from loguru import logger
from os.path import join
from torch.nn import functional as F
from torchvision import transforms
import argparse
import numpy as np
import torch


from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from style_transfer import run_alignment
from util import save_image, load_image

DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     DEVICE = torch.device("mps")
# print(f"Device for running: {DEVICE}")
print(50 * "-")


class TestOptions:
    def __init__(self, options):
        self.options = options

        self.parser = argparse.ArgumentParser(
            description="Exemplar-Based Style Transfer"
        )

        self.parser.add_argument(
            "--style", type=str, default="pixar", help="target style type"
        )
        self.parser.add_argument(
            "--style_id", type=int, default=13, help="the id of the style image"
        )
        self.parser.add_argument(
            "--truncation",
            type=float,
            default=0.75,
            help="truncation for intrinsic style code (content)",
        )
        self.parser.add_argument(
            "--weight",
            type=float,
            nargs=18,
            default=[0.6] * 11 + [1] * 7,
            help="weight of the extrinsic style",
        )
        self.parser.add_argument(
            "--name",
            type=str,
            default="arcane_transfer",
            help="filename to save the generated images",
        )
        self.parser.add_argument(
            "--preserve_color",
            action="store_true",
            default=True,
            help="preserve the color of the content image",
        )
        self.parser.add_argument(
            "--model_path",
            type=str,
            default=join(CURR_PATH, "checkpoint"),
            help="path of the saved models",
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="generator.pt",
            help="name of the saved dualstylegan",
        )
        self.parser.add_argument(
            "--output_path",
            type=str,
            default="./output/",
            help="path of the output images",
        )
        self.parser.add_argument(
            "--data_path", type=str, default="./data/", help="path of dataset"
        )
        self.parser.add_argument(
            "--align_face",
            action="store_true",
            default=True,
            help="apply face alignment to the content image",
        )
        self.parser.add_argument(
            "--exstyle_name",
            type=str,
            default=None,
            help="name of the extrinsic style codes",
        )

    def parse(self):
        self.opt, unkown = self.parser.parse_known_args()
        if self.opt.exstyle_name is None:
            if os.path.exists(
                os.path.join(
                    self.opt.model_path, self.opt.style, "refined_exstyle_code.npy"
                )
            ):
                self.opt.exstyle_name = "refined_exstyle_code.npy"
            else:
                self.opt.exstyle_name = "exstyle_code.npy"

        args = vars(self.opt)
        # print(f"Load options: {self.options}")
        for name, value in sorted(args.items()):
            if name in dict(self.options).keys():
                setattr(self.opt, name, self.options[name])

            # print("%s: %s" % (str(name), str(self.opt.name)))

        return self.opt


def my_run_alignment2(img):
    import dlib
    from model.encoder.align_all_parallel import align_face, my_align_face

    modelname = os.path.join(
        join(CURR_PATH, "checkpoint"), "shape_predictor_68_face_landmarks.dat"
    )
    if not os.path.exists(modelname):
        import wget, bz2

        wget.download(
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            modelname + ".bz2",
        )
        zipfile = bz2.BZ2File(modelname + ".bz2")
        data = zipfile.read()
        open(modelname, "wb").write(data)

    predictor = dlib.shape_predictor(modelname)
    aligned_image = my_align_face(img, predictor=predictor)
    return aligned_image


isInited = False
transform = None
encoder = None


def load_encoder(encoder_path):
    ckpt = torch.load(encoder_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = encoder_path
    opts = Namespace(**opts)
    opts.device = DEVICE
    encoder = pSp(opts)
    encoder.eval()
    encoder = encoder.to(DEVICE)
    print(f"[Cartoonize] Loaded model to GPU!")


def Init():
    global isInited, encoder, DEVICE, transform
    if isInited:
        return
    isInited = True
    logger.info("Initializing...")
    transform = transforms.Compose(
        [
            # transforms.Resize(size=1024),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # if encoder is None:
    #     print("Initializing encoder...")
    #     model_path = os.path.join(CURR_PATH, 'checkpoint', "encoder.pt")
    #     # ckpt = torch.load(model_path, map_location='cpu')
    #     ckpt = torch.load(model_path, map_location=DEVICE)

    #     opts = ckpt["opts"]
    #     opts["checkpoint_path"] = model_path
    #     opts = Namespace(**opts)
    #     opts.device = DEVICE
    #     encoder = pSp(opts)
    #     encoder.eval()
    #     encoder = encoder.to(DEVICE)


def StyleTransfer(I, image_size=1024, options=None):
    global transform, encoder, DEVICE

    device = DEVICE
    # print("DEVICE: ", device)
    parser = TestOptions(options)
    args = parser.parse()

    # print(f"Args: {args}")
    # print("*" * 98)

    # transform = transforms.Compose([
    #     # transforms.Resize(size=1024),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # print(f'{datetime.now().strftime("%H_%M_%S")}')

    generator = DualStyleGAN(image_size, 512, 8, 2, res_index=6)
    generator.eval()

    style_model_path = os.path.join(args.model_path, args.style, args.model_name)
    # print(f"style_model_path: {style_model_path}")

    ckpt = torch.load(style_model_path, map_location="cpu")
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)
    # print(f'{datetime.now().strftime("%H_%M_%S")}')

    encoder = None
    if "encoder" in options and options["encoder"] is not None:
        encoder = options["encoder"]
    else:
        model_path = os.path.join(args.model_path, "encoder.pt")
        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = model_path
        opts = Namespace(**opts)
        opts.device = device
        encoder = pSp(opts)
        encoder.eval()
        encoder = encoder.to(device)

    logger.info(f"Encoder loaded.")

    exstyles = np.load(
        os.path.join(args.model_path, args.style, args.exstyle_name),
        allow_pickle="TRUE",
    ).item()

    logger.info("Load models successfully!")
    with torch.no_grad():
        viz = []
        # load content image
        # if args.align_face:
        #     I = transform ol2d(I, 1024)
        # else:
        #     I = load_image(args.content).to(device)
        viz += [I]

        # reconstructed content image and its intrinsic style code
        img_rec, instyle = encoder(
            F.adaptive_avg_pool2d(I, 256),
            randomize_noise=False,
            return_latents=True,
            z_plus_latent=True,
            return_z_plus_latent=True,
            resize=False,
        )
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        viz += [img_rec]

        stylename = list(exstyles.keys())[args.style_id]
        latent = torch.tensor(exstyles[stylename]).to(device)
        if args.preserve_color:
            latent[:, 7:18] = instyle[:, 7:18]
        # extrinsic styte code
        exstyle = generator.generator.style(
            latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])
        ).reshape(latent.shape)

        # load style image if it exists
        S = None
        if os.path.exists(
            os.path.join(args.data_path, args.style, "images/train", stylename)
        ):
            S = load_image(
                os.path.join(args.data_path, args.style, "images/train", stylename)
            ).to(device)
            viz += [S]

        # style transfer
        # input_is_latent: instyle is not in W space
        # z_plus_latent: instyle is in Z+ space
        # use_res: use extrinsic style path, or the style is not transferred
        # interp_weights: weight vector for style combination of two paths
        img_gen, _ = generator(
            [instyle],
            exstyle,
            input_is_latent=False,
            z_plus_latent=True,
            truncation=args.truncation,
            truncation_latent=0,
            use_res=True,
            interp_weights=args.weight,
        )
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        viz += [img_gen]

    # torch.cuda.empty_cache()
    # print("Generate images successfully!")
    # print(f'{datetime.now().strftime("%H_%M_%S")}')

    # save_name = (
    #     args.name
    # )  # +'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])
    # print(save_name)
    # save_image(
    #     torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256),
    #                                 4, 2).cpu(),
    #     os.path.join(args.output_path, save_name + '_overview.jpg'))

    # print("Save images successfully!")
    return img_gen[0].cpu()


# function return current time with seconds
def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")


def process_cropped_face(image_path, cropped_image_path):
    I = run_alignment(
        {"content": image_path, "model_path": join(CURR_PATH, "checkpoint")}
    )
    I.convert("RGB")
    I.save(cropped_image_path)
    # save_image(I, cropped_image_path)
    return cropped_image_path


def StyleTransferService(image, options=None, is_save=False):
    logger.info(f"StyleTransferService start...")
    # print(f"Params: {options} \n\n")

    Init()
    # image_size = 720

    if type(image) != str:
        # type of image is tensor image or object
        I = transform(my_run_alignment2(image)).unsqueeze(dim=0).to(DEVICE)
        I = F.adaptive_avg_pool2d(I, 1024)
    else:
        # type of image is string
        I = (
            transform(
                run_alignment(
                    {"content": image, "model_path": join(CURR_PATH, "checkpoint")}
                )
            )
            .unsqueeze(dim=0)
            .to(DEVICE)
        )
        I = F.adaptive_avg_pool2d(I, 1024)

    logger.info(f"Starting cartoonize...")

    outputImage = StyleTransfer(I, image_size=1024, options=options)

    if is_save:
        """
        This is used to check whether image processed or not!
        output_path = f"{CARTOONIZE_RESOURCE}/output/{params.request_id}_{params.style}_{params.style_id}.jpg"

        """
        # image_path = f"{CURR_PATH}/output/out_{current_time}.jpg"
        image_path = options["output_path"]
        save_image(outputImage, image_path)
        logger.info(f"Saved at {image_path}")

    logger.info(f"Finished!")

    return outputImage


if __name__ == "__main__":
    # torch.cuda.empty_cache()
    print(f'{datetime.now().strftime("%H_%M_%S")}')

    img = Image.open("i.jpg")
    print("****")
    # load_image("i.jpg")
    StyleTransferService(img, None, True)
    # print(f'{datetime.now().strftime("%H_%M_%S")}')
