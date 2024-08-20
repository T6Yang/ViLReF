
class feat_extract_img():
    def __init__(self, model):
        self.model = model

    def __call__(self, imgs):
        return self.model.module.encode_image_featExt(imgs)

        