"""YOLO_v3 Model Defined in Keras.
建立Darknet网络"""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D.
    DBL:darknet网络的最基本结构，卷积层+归一化层+激活函数leaky relu"""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}  #进行了L2正则化
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same' #根据valid的值确定是否进行步长为2的卷积
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    输入为卷积后的输出"""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D
    darknet网络的基础结构之一，res=DBL+DBL+上一次的残差（res）
    resblock=补零+DBL+res*n'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x) #在行的最前和最后都增加1行0
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):#darknet的整体结构
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):#第一个y1输出
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras.
    y1,y2,y3输出"""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras
    tiny_yolo的结构，有池化，yolov3无池化，通过2的步长实现'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters.
    全连接层的操作，feats参数, anchors框, num_classes种类, input_shape输入, calc_loss=False
    输出了xy，wh，置信度以及预测种类，所谓的端到端输出,把darknet输出的值跟我们的y_true对应上,feat 是某一个尺度下的特征图 如13*13*255"""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])#张量扩展
    grid = K.concatenate([grid_x, grid_y])#x，y组合
    grid = K.cast(grid, K.dtype(feats))#将grid转换为feats的格式
    # 这里输出如果是小尺度就是 13*13*num_anchor*（num_classes+5）
    # num_anchor 是3 表示有三个anchor 负责这层 也表示每个grid预测3个bounding box
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # 调整预置到每个空间网格点和锚的大小
    # Adjust preditions to each spatial grid point and anchor size.
    # 这里中心坐标x ，y和宽高wh的转换可以看
    # 把特征图的输出转换成 相对于特征图的比例 和坐标相对于整张图像的比例是等价的

    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # 把宽高 wh转换成相对于整张图像的比例
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    # 转换置信度和类别概率
    box_class_probs = K.sigmoid(feats[..., 5:])
    # 如果计算loss 为真 返回grid feats box_xy box_wh
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs
    #此时的各个值是经过了归一化等处理之后的


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes
    输出正确的boxes参数值'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes.
    yolo_outputs,yolo输出
              anchors,框
              num_classes,种类
              image_shape,
              max_boxes=20,最多的候选框数量
              score_threshold=.6,置信度阈值
              iou_threshold=.5iou预祝"""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting采用yolo或者tiny
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)#得到每个框的值
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])#boolean_mask(a,b) 将使a (m维)矩阵仅保留与b中“True”元素同下标的部分
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)#非最大值抑制=NMS
        class_boxes = K.gather(class_boxes, nms_index)#从’params"中，按照axis坐标和indices标注的元素下标，把这些元素抽取出来组成新的tensor.
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c#新建一个与class_box_scores大小一致的矩阵，乘以c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    # 检查有无异常数据 即txt提供的box id 是否存在大于 num_class的情况
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'#强行设定
    num_layers = len(anchors)//3 # default setting#判断是yolo还是tiny
    # 不同尺度的anchor 分到不同尺度的 输出
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2#取中心点
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    #现在true_boxes 中的数据成了 x,y,w,h

    #这个m应该是batch的大小 即是输入图片的数量
    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)#扩展数组的形状
    anchor_maxes = anchors / 2.#网格中心为原点（即网格中心坐标为 （0,0） ）,　计算出anchor 右下角坐标
    anchor_mins = -anchor_maxes #网格中心为原点 计算anchorr 左上角坐标
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):#NMS，选取最佳框
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.# 假设　bouding box 的中心也位于网格的中心
        box_mins = -box_maxes

        # 下面就是在计算 ground_true与anchor box的交并比
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        #对于每个真实box 找到最匹配的anchor best_anchor 的格式为 bounding_box id -> anchor_id
        best_anchor = np.argmax(iou, axis=-1)
        #遍历所有 匹配的anchor
        for t, n in enumerate(best_anchor):
            #遍历anchor 尺寸 3个尺寸
            #因为此时box 已经和一个anchor box匹配上，看这个anchor box属于那一层，小，中，大，然后将其box分配到那一层
            for l in range(num_layers):
                #anchor_mask [ 6,7,8   3,4，5     0，1,2]
                #如果匹配的这个n即 anchor id在 l这一层，那么进行赋值数据
                if n in anchor_mask[l]:
                    # np.floor 返回不大于输入参数的最大整数。 即对于输入值 x ，将返回最大的整数 i ，使得 i <= x。
                    # true_boxes x,y,w,h, 此时x y w h都是相对于整张图像的
                    # 第b个图像 第 t个 bounding box的 x 乘以 第l个grid shap的x（grid shape 格式是hw，
                    # 因为input_shape格式是hw）
                    # 找到这个bounding box落在哪个cell的中心
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    #找到n 在 anchor_box的索引位置
                    k = anchor_mask[l].index(n)
                    #得到box的id
                    c = true_boxes[b,t, 4].astype('int32')
                    # 第b个图像 第j行 i列 第k个anchor x，y，w，h,confindence,类别概率
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1#置信度是1 因为含有目标
                    y_true[l][b, j, i, k, 5+c] = 1#类别的one-hot编码

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)
（1）args包括两部分，第一部分是*model_body.output，就是三组 (batchsize, grid, grid, 75)的darknet输出；第二部分是*y_true，就是上一篇文章咱们说到的 三组(batchsize, grid, grid, 3, 25)的Y真实值

（2）anchors就是[[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]9组anchors

（3）num_classes=20（COCO数据集为80）

（4）ignore_thresh指的是iou的最小达标值
    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]#这里存储的是输出
    y_true = args[num_layers:]#这里存储的是ground truth
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]#置信度
        true_class_probs = y_true[l][..., 5:]# 类别概率
        # 这个yolo_head  因为calc_loss=True 返回 grid 特征图 xy wh，特征图是最原始输出，xy是相对于特征图，wh是相对于整张图像
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        #将xy与wh进行拼接
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        # #darknet 原始盒子 来计算损失
       #将y_ture转换成最原始 的 没有加经过处理的输出 是yolo_head函数中转换xy wh的逆过程
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        #xy true一开始存储的是xy相对于整张图像的比例值大小 经过操作后 就变成 相对于相对于当前cell的偏移值了
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]
        """
        大框给小权重，小框给大权重，因为大框的xywh不需要学得那么好，而小框则对xywh很敏感

        为了调整不同大小的预测框所占损失的比重，真值框越小，
        box_loss_scale越大，这样越小的框的损失占比越大，和v1，v2里采用sqrt(w)的目的一样
        """

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        #object 4:5 存储的是置信度 将其转换为bool类型
        def loop_body(b, ignore_mask):
            # 这里看了下tf.boolean_mast 函数 将置信度为1 （即含有目标） 赋值给true_box
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            #遍历第b（即mini_batch_size）个图像 这个图像上所有的预测box和 当前尺度下的所有gt做iou
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            #如果一张图片的最大iou 都小于阈值 认为这张图片没有目标
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        """
        如果某个anchor不负责预测GT，且该anchor预测的框与图中所有GT的IOU都小于某个阈值，则让它预测背景，
        如果大于阈值则不参与损失计算
        """
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)
        # K.binary_crossentropy is helpful to avoid exp overflow.
        # 现在raw_true_xy，raw_pred 都是特征图得直接输出 没有经过任何处理
        # 这里会对raw_pred进行sigmod操作 所以xyloss 输入交叉熵的都是相对于 当前cell左上角偏移值 的 一个交叉熵
        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
#object_mask就是置信度
#box_loss_scale可以理解为2-w*h
#raw_true_xy就是真实的xy坐标点了
#raw_pred[..., :2]是xy预测坐标点
        """
        1-object_mast 说明该这个anchor 不负责 预测GT Object_mask=y_ture 可以再仔细看下y_true的存储格式
          如果某个anchor不负责预测GT，且该anchor预测的框与图中所有GT的IOU都小于某个阈值，则让它预测背景，
        如果大于阈值则不参与损失计算
        """
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
