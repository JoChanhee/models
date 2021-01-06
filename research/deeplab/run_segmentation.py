import os
import sys
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import png

# %tensorflow_version 1.x
import tensorflow as tf
import cv2


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  # TODO :: Need to modify color map with labels
  for idx, color in enumerate(colormap):
    colormap[idx] = [255, 255, 255]
    # colormap[idx] = [0, 0, 0]

  colormap[15] = [42, 254, 1]


  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')


  return colormap[label]


def vis_segmentation(image, seg_map, target_path):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  # plt.subplot(grid_spec[0])
  # plt.imshow(image)
  # plt.axis('off')
  # plt.title('input image')


  # for y, y_map in enumerate(seg_map):
  #   for x, x_map in enumerate(y_map):
  #     if seg_map[y][x] != 0:
  #       seg_map[y][x] = 1
  #
  # mask_map = np.int8(seg_map)


  # cv2.imwrite(IMAGE_URL + "_map.png", mask_map)

  # png.from_array(mask_map, mode="L").save("map.png")

  # plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  # plt.imshow(seg_image)
  # plt.axis('off')
  # plt.title('segmentation map')


  # plt.subplot(grid_spec[2])
  # plt.imshow(image)
  # plt.imshow(seg_image, alpha=0.7)
  # plt.axis('off')
  # plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)

  for idx, label in enumerate(unique_labels):
    if label != 0:
      unique_labels[idx] = 1

  # ax = plt.subplot(grid_spec[3])
  # plt.imshow(
  #     FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  # ax.yaxis.tick_right()
  # plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  # plt.xticks([], [])
  # ax.tick_params(width=0.0)
  # plt.grid('off')


  mix = 0.5
  image_np = np.asarray(image, dtype="uint8")

  # scharrx = cv2.Scharr(image_np, -1, 1, 0)
  # scharry = cv2.Scharr(image_np, -1, 0, 1)
  #
  # sobelx = cv2.Sobel(image_np, -1, 1, 0, ksize=3)
  # sobely = cv2.Sobel(image_np, -1, 0, 1, ksize=3)
  # merged2 = np.hstack((sobelx, sobely, sobelx + sobely))
  #
  # merged = np.hstack((scharrx, scharry))
  # edge_image = cv2.Laplacian(image_np, -1)
  # edge_image = cv2.Canny(image_np, 100, 200)
  # plt.imshow(merged2)
  # plt.show()


  # weighted_img = cv2.addWeighted(image_np, mix, seg_image, 1.0 - mix, 0)
  weighted_img = cv2.add(image_np, seg_image)

  image_np.setflags(write=1)
  image_np[np.where((seg_image != [255, 255, 255]).any(axis=-1))] = [255, 102, 51]

  # plt.imshow(image_np)
  # plt.show()


  # im = Image.fromarray(seg_image)
  im = Image.fromarray(image_np)
  im.save(target_path)



  # plt.savefig(target_path)
  # plt.show()


LABEL_NAMES = np.asarray([
  'background',
  'person'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

# TODO :: Need to modify model name using external parameters
MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = MODEL_NAME + '.tar.gz' # 'deeplab_model.tar.gz'


# TODO :: Check if there is a model or not
# model_dir = tempfile.mkdtemp()
# tf.gfile.MakeDirs("model")
#
download_path = os.path.join("model", _TARBALL_NAME)
# print('downloading model, this might take a while...')
# urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
#                    download_path)
# print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')


SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
# IMAGE_URL = 'http://photo.jtbc.joins.com/news/2019/01/08/201901081539178857.jpg'  #@param {type:"string"}

_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
               'deeplab/g3doc/img/%s.jpg?raw=true')



def get_separated_segmap_with_hip_skeleton(seg_map, skeleton_point, is_top=True):

  height = seg_map.shape[0]
  target_y = skeleton_point[1]

  target_height = int(float(height) * target_y)

  if is_top:
    seg_map[target_height:] = 0
  else:
    seg_map[:target_height] = 0

  return seg_map



directions = [(-1,0), (0,1), (1,0), (0,-1)]
HIGHLIGHT = 222
VISITED = 100
def highlight_inner_dfs(x:int, y:int, seg_map):
  width = seg_map.shape[1]
  height = seg_map.shape[0]

  for direction in directions:
    dx = x + direction[0]
    dy = y + direction[1]
    ddx = dx + direction[0]
    ddy = dy + direction[1]

    if dx >= 0 and dy >= 0 and dx < width and dy < height \
            and seg_map[dy][dx] != 0 and seg_map[dy][dx] != VISITED and seg_map[dy][dx] != HIGHLIGHT:

      if ddx >= 0 and ddy >= 0 and ddx < width and ddy < height and seg_map[ddy][ddx] != VISITED and seg_map[ddy][ddx] == 0:
        seg_map[dy][dx] = HIGHLIGHT
      else:
        seg_map[dy][dx] = VISITED
        highlight_inner_dfs(dx, dy, seg_map)



def get_highlight_segmap_inner(seg_map, hip_point):
  # sys.setrecursionlimit(5000)
  start_point = (0, 0)

  if hip_point:
    width = seg_map.shape[1]
    height = seg_map.shape[0]
    target_x = hip_point[0]
    target_y = hip_point[1]

    target_width = int(float(width) * target_x)
    target_height = int(float(height) * target_y)
    start_point = (target_width+1, target_height+1)

  highlight_inner_dfs(start_point[0], start_point[1], seg_map)

  return seg_map



def get_highlight_segmap(seg_map, hip_point):
  seg_map = seg_map.astype('uint8')
  k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
  gradient_map = cv2.morphologyEx(seg_map, cv2.MORPH_GRADIENT, k)


  seg_map = seg_map.astype('int64')
  gradient_map = gradient_map.astype('int64')


  if hip_point:
    width = seg_map.shape[1]
    height = seg_map.shape[0]
    target_x = hip_point[0]
    target_y = hip_point[1]

    target_width = int(float(width) * target_x)
    target_height = int(float(height) * target_y)

    gradient_map[target_height-5 : target_height+5] = 0


  # # debug images
  # segmap_image = label_to_color_image(seg_map).astype(np.uint8)
  # gradient_image = label_to_color_image(gradient_map).astype(np.uint8)
  #
  # merged = np.hstack((segmap_image, gradient_image))
  # cv2.imshow('gradient', merged)
  # cv2.waitKey(0)

  return gradient_map



# get normalized hip point of original entire image (normalized point of box -> normalized point of image)
def get_normalized_hip_point_of_image(original_img, skeleton, skeleton_box):

  # get center normalized hip point
  left_hip_point = skeleton[11]
  right_hip_point = skeleton[12]
  center_hip_point = (left_hip_point + right_hip_point) * 0.5

  # get absolute hip point
  left_top_box_point = skeleton_box[0]
  right_bottom_box_point = skeleton_box[1]

  absolute_hip_x = (right_bottom_box_point[0] - left_top_box_point[0]) * center_hip_point[0] + left_top_box_point[0]
  absolute_hip_y = (right_bottom_box_point[1] - left_top_box_point[1]) * center_hip_point[1] + left_top_box_point[1]

  normalized_hip_x = absolute_hip_x / original_img.width
  normalized_hip_y = absolute_hip_y / original_img.height

  return (normalized_hip_x, normalized_hip_y)



def run_visualization(url, target_path, skeleton_point=None, skeleton_box=None):
  """Inferences DeepLab model and visualizes result."""
  # TODO :: Need to modify url to path
  try:
    # f = urllib.request.urlopen(url)
    # jpeg_str = f.read()
    # original_im = Image.open(BytesIO(jpeg_str))


    original_im = Image.open(url)
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(original_im)


  if skeleton_point is not None:
    normalized_hip_point = get_normalized_hip_point_of_image(original_im, skeleton_point, skeleton_box)

    seg_map = get_separated_segmap_with_hip_skeleton(seg_map, normalized_hip_point, is_top=True)

    seg_map = get_highlight_segmap(seg_map, normalized_hip_point)

  vis_segmentation(resized_im, seg_map, target_path)



# image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE


# TODO :: Need to modify video paths using external parameters
names = ['wannabe']

for name in names:

  target_video = 'data/' + name + '_input.mp4'
  frame_dir = 'data/' + name + '_frames'
  output_dir = 'output/' + name + '_frames_top'
  video_name = 'output/' + name + '_segmentation_top.mp4'
  skeleton_name = 'data/' + name + '_skeleton.npy'
  skeleton_box = 'data/' + name + '_box.npy'

  if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # 1. Frame extraction
  vidcap = cv2.VideoCapture(target_video)
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
  success,image = vidcap.read()
  count = 0
  # while success:
  #   cv2.imwrite(os.path.join(frame_dir, "frame%05d.jpg" % count), image)     # save frame as JPEG file
  #   success,image = vidcap.read()
  #   if count % 60 == 0:
  #     print('Read a new frame: ', success, " / ", str(count))
  #   count += 1

  # 2. Inference multiple images
  # 2-1. Load skeleton numpy
  # skeleton_np = np.load(skeleton_name)
  # skeleton_box_np = np.load(skeleton_box)
  #
  # for frame_num, filename in enumerate(os.listdir(frame_dir)):
  #   file_fullpath = os.path.join(frame_dir, filename)
  #
  #   run_visualization(url=file_fullpath,
  #                     target_path=os.path.join(output_dir, os.path.splitext(filename)[0]) + '_output.jpg',
  #                     skeleton_point=skeleton_np[frame_num], skeleton_box=skeleton_box_np[frame_num])


  # 3. Make video with output frames
  images = os.listdir(output_dir)
  images.sort()
  frame = cv2.imread(os.path.join(output_dir, images[0]))
  height, width, layers = frame.shape

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # fourcc = cv2.VideoWriter_fourcc(*'X264')
  # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

  for image in images:
      video.write(cv2.imread(os.path.join(output_dir, image)))

  cv2.destroyAllWindows()
  video.release()