import cv2
import numpy as np

from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer


colors = {
    'blue': [0.7, 0.7, 0.9],
    'green': [0.7, 0.9, 0.7],
    'red': [0.9, 0.7, 0.7]
}


class SMPLRenderer(object):
    def __init__(self, img_size=224, focal=500., princpt=[],
                 faces=None, face_path="./model_files/smpl_faces.npy"):
        self.faces = np.load(face_path) if faces is None else faces
        self.w = img_size[1]
        self.h = img_size[0]
        self.focal = focal
        self.princpt = princpt

    def __call__(self, verts, princpt=None, img=None,
                 do_alpha=False, far=None, near=None, color_id=0,
                 img_size=None, cam_rt=np.zeros(3), cam_t=np.zeros(3)):

        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        if princpt is None:
            princpt = [w / 2, h / 2]
        else:
            princpt = self.princpt

        use_cam = ProjectPoints(
            f=self.focal, rt=cam_rt, t=cam_t, k=np.zeros(5), c=princpt)

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)

        imtmp = render_model(verts, self.faces, w, h, use_cam, 
                             do_alpha=do_alpha, img=img, far=far, near=near, color_id=color_id)

        return (imtmp * 255).astype('uint8')


def render_model(verts, faces, w, h, cam,
                 near=0.5, far=25, img=None, do_alpha=False, color_id=0):
    
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    color_list = list(colors.values())
    color = color_list[color_id % len(color_list)]

    rendered_img = simple_renderer(rn, verts, faces, color=color)

    # If white bg, make transparent.
    if img is None and do_alpha:
        rendered_img = get_alpha(rendered_img)

    elif img is not None and do_alpha:
        rendered_img = append_alpha(rendered_img)

    return rendered_img


def _create_renderer(w=640, h=480, rt=np.zeros(3), t=np.zeros(3),
                     f=None, c=None, k=None, near=.5, far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}

    return rn


def simple_renderer(rn, verts, faces, color, yrot=np.radians(120)):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))
    
    return rn.r


def _rotateY(points, angle):
    """ Rotate the points by a specified angle. """
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    
    return np.dot(points, ry)


def get_alpha(img, bgval=1.):
    alpha = (~np.all(img == bgval, axis=2)).astype(img.dtype)

    b_channel, g_channel, r_channel = cv2.split(img)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha.astype(img.dtype)))

    return im_RGBA


def append_alpha(img):
    alpha = np.ones_like(img[:, :, 0]).astype(img.dtype)
    if np.issubdtype(img.dtype, np.uint8):
        alpha = alpha * 255
    b_channel, g_channel, r_channel = cv2.split(img)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha))

    return im_RGBA