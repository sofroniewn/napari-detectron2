# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import warnings
import numpy as np
import matplotlib.colors as mplc
import napari


from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer as Detectron2Visualizer


logger = logging.getLogger(__name__)

__all__ = ["ColorMode", "VisImage", "Visualizer"]


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self.viewer = None
        self._setup_viewer(img)

    def _setup_viewer(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.
        """
        # Create the napari viewer
        self.viewer = napari.Viewer()
        scale = [self.scale] * 2
        self.viewer.add_image(img, scale=scale, name='input')
        self.viewer.add_shapes(scale=scale, name='annotations')
        self.viewer.add_points(size=1, edge_width=0, scale=scale, name='text')
        self.viewer.layers['text'].text._mode = 'formatted'
        self.viewer.layers['text'].text.anchor = 'upper_left'

    @property
    def img(self):
        """array: Image data."""
        return self.viewer.layers['input'].data

    @img.setter
    def img(self, data):
        self.viewer.layers['input'].data = data

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.viewer.reset_view()
        self.viewer.screenshot(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
        """
        self.viewer.reset_view()
        img_rgba = self.viewer.screenshot()
        rgb, _ = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer(Detectron2Visualizer):
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.

    To obtain a consistent style, you can implement custom drawing functions with the
    abovementioned primitive methods instead. If you need more customized visualization
    styles, you can process the data yourself following their format documented in
    tutorials (:doc:`/tutorials/models`, :doc:`/tutorials/datasets`). This class does not
    intend to satisfy everyone's preference on drawing styles.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        super().__init__(img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode)
        # Overwrite matplotlib output with napari output
        self.output = VisImage(self.img, scale=scale)

    def show(self):
        """
        Show the napari viewer.
        """
        napari.run()

    """
    Overwrite primitive drawing functions:
    """

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        # if horizontal_alignment != "center":
        #     warnings.warn('Non-default horizontal alignment styles are not supported, use "center"')

        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        x, y = position
        # Note color, size etc are same for all text
        self.output.viewer.layers['text'].text.color = color
        self.output.viewer.layers['text'].text.size = 2 * font_size * self.output.scale
        self.output.viewer.layers['text'].text.rotation = rotation
        self.output.viewer.layers['text'].add([y, x])
        values = self.output.viewer.layers['text'].text.values
        new_values = np.concatenate((values, [text]), axis=0)
        self.output.viewer.layers['text'].text.values = new_values

        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        if line_style != "-":
            warnings.warn('Non-default line styles are not supported, use "-"')

        x0, y0, x1, y1 = box_coord
        linewidth = max(self._default_font_size / 4, 1)

        # Note opacity is gloabal for all shapes, could make shape specific if desirable
        self.output.viewer.layers['annotations'].opacity = alpha
        self.output.viewer.layers['annotations'].add(
            [[(y0, x0), (y1, x1)]],
            shape_type='rectangle',
            face_color=[0, 0, 0, 0],
            edge_color=edge_color,
            edge_width=linewidth * self.output.scale,
        )
        return self.output

    def draw_rotated_box_with_label(
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None
    ):
        """
        Draw a rotated box with label on its top-left corner.

        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
        self.draw_box(rotated_rect, alpha=alpha, edge_color=edge_color, line_style=line_style)

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.height * self.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.viewer.layers['annotations'].add(
            [[(y, x), (radius, radius)]],
            shape_type='rectangle',
            face_color=color,
            edge_width=0,
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linestyle != "-":
            warnings.warn('Non-default line styles are not supported, use "-"')

        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        line_data = [[y, x] for x, y in zip(x_data, y_data)]
        self.output.viewer.layers['annotations'].add(
            [line_data],
            shape_type='path',
            edge_color=color,
            edge_width=linewidth * self.output.scale,
        )
        return self.output

    def draw_binary_mask(
        self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=0
    ):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn in the object's center of mass.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component small than this will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        warnings.warn('draw_binary_mask is not supported')

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        linewidth = max(self._default_font_size // 15 * self.output.scale, 1)

        polygon_data = segment[:, [1, 0]]
        self.output.viewer.layers['annotations'].add(
            [polygon_data],
            shape_type='polygon',
            face_color=mplc.to_rgb(color) + (alpha,),
            edge_color=edge_color,
            edge_width=linewidth,
        )
        return self.output