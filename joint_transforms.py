import numbers
import random

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, ins_):
        if random.random() < 0.5:
            outs_ = []
            for in_ in ins_:
                outs_.append(in_.transpose(Image.FLIP_LEFT_RIGHT))
            return outs_
        else:
            return ins_


class RandomVerticalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, ins_):
        outs_ = []
        for in_ in ins_:
            outs_.append(in_.resize((self.size, self.size), Image.BILINEAR))
        return outs_


class RandomFlipRotate(object):
    def __init__(self):
        self.degrees = [0, 90, 180, 270]

    def __call__(self, img, mask):
        degree = random.choice(self.degrees)
        if degree == 0:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img, mask
        elif degree == 90:
            img = img.rotate(degree, Image.BILINEAR)
            mask = mask.rotate(degree, Image.NEAREST)
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                return img, mask
        elif degree == 180:
            img = img.rotate(degree, Image.BILINEAR)
            mask = mask.rotate(degree, Image.NEAREST)
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img, mask
        elif degree == 270:
            img = img.rotate(degree, Image.BILINEAR)
            mask = mask.rotate(degree, Image.NEAREST)
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                return img, mask

