import os
import sys
import logging
import datetime
import numpy as np
import imageio
import json
import random
import time
import tensorflow as tf
from run_nerf_helpers import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.enable_eager_execution()

# 配置日志记录器
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)


def batchify(fn, chunk):
    """构造一个适用于小批次的版本的'fn'函数。
    @param fn: 输入函数，用于处理小批次数据。
    @param chunk: 小批次大小。
    @return: 新的函数，可以逐个处理小批次数据。
    """
    if chunk is None:
        return fn  # 如果chunk为None，则返回原始函数fn。

    def ret(inputs):
        # 将输入数据按照给定的chunk分割成多个小批次，逐个应用fn，最后将结果合并。
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret  # 返回新函数ret。



def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """准备输入并应用网络'fn'。
    @param inputs: 输入张量，形状为 [batch_size, ...]。
    @param viewdirs: 视角方向张量，形状为 [batch_size, 3] 或 None。
    @param fn: 网络函数，接受嵌入向量作为输入，输出处理结果。
    @param embed_fn: 输入嵌入函数，将输入张量嵌入到一个新的空间中。
    @param embeddirs_fn: 视角方向嵌入函数，将视角方向嵌入到同一空间中。
    @param netchunk: 批次大小，用于将嵌入向量分成多个小批次进行计算。默认值为1024*64。
    @return: 输出张量，形状与输入张量相同。
    """
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])  # 将输入数据展平为二维张量。
    # 将展平后的输入数据嵌入到一个新的空间中。
    embedded = embed_fn(inputs_flat)
    # 如果视角方向不为空，则将视角方向广播到与输入张量相同的形状，并将其嵌入到同一空间中。
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)
    # 将嵌入向量分成多个小批次，逐个应用网络函数fn，最后将结果合并。
    outputs_flat = batchify(fn, netchunk)(embedded)
    # 将输出结果重新整形为与输入张量相同的形状。
    outputs = tf.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs



def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False):
    """体积渲染。Volumetric rendering.
    @param ray_batch: 数组，形状为[batch_size, ...]。沿着射线进行采样所需的所有信息，包括：射线起点、射线方向、最小距离、最大距离和单位幅度的观察方向。
    @param network_fn: 函数。用于预测空间中每个点的RGB和密度的模型。
    @param network_query_fn: 用于向network_fn传递查询的函数。
    @param N_samples: 整数。沿每条射线采样的不同时间数量。
    @param retraw: 布尔值。如果为True，则包括模型的原始、未处理的预测结果。
    @param lindisp: 布尔值。如果为True，则沿着深度的倒数进行线性采样，而不是沿深度采样。
    @param perturb: 浮点数，取值为0或1。如果非零，则每条射线将在分层随机时间点进行采样。
    @param N_importance: 整数。每条射线额外采样的时间次数。这些样本只会传递给network_fine。
    @param network_fine: “细化”网络，与network_fn具有相同的规范。
    @param white_bkgd: 布尔值。如果为True，则假定白色背景。
    @param raw_noise_std: ……
    @param verbose: 布尔值。如果为True，则打印更多的调试信息。

    @return Returns:
    @return rgb_map: [num_rays, 3]。一条射线的估计RGB颜色。来自细模型。
    @return disp_map: [num_rays]。视差图。深度的倒数。
    @return acc_map: [num_rays]。沿每条射线累积的不透明度。来自细模型。
    @return raw: [num_rays, num_samples, 4]。模型的原始预测。
    @return rgb0: 请参见rgb_map。粗模型的输出。
    @return disp0: 请参见disp_map。粗模型的输出。
    @return acc0: 请参见acc_map。粗模型的输出。
    @return z_std: [num_rays]。每个样本的距离沿着射线的标准差。
    """

    def raw2outputs(raw, z_vals, rays_d):
        """将模型的预测值转换为语义上有意义的值。
        @param raw: [num_rays, num_samples along ray, 4]。模型的预测。
        @param z_vals: [num_rays, num_samples along ray]。集成时间。
        @param rays_d: [num_rays, 3]. 每条射线的方向。
        @return rgb_map: [num_rays, 3]. 一条射线的估计RGB颜色。
        @return disp_map: [num_rays]. 失谐图。逆深度图。
        @return acc_map: [num_rays]. 沿每条射线累积的不透明度。来自细模型。
        @return weights: [num_rays, num_samples]. 分配给每个采样颜色的权重。
        @return depth_map: [num_rays]. 估计的距离到对象。
        """
        # 用于从模型预测中计算密度的函数。这个值严格在[0, 1]之间。
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # 计算每个集成时间沿射线的“距离”（时间）。
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # 最后一个集成时间的“距离”为无穷大。
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # 每个距离乘以其相应射线方向的范数，以将其转换为真实世界距离（考虑非单位方向）。
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # 提取每个采样位置沿每个射线的RGB。
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # 为密度的每个采样位置沿每个射线预测密度。较高的值意味着在这一点被吸收的可能性更高。
        # raw_noise_std是添加到密度的模型预测的噪声的标准差。可用于训练期间正则化网络（防止浮动伪像）。
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # 计算沿每条射线的每个采样点的RGB权重。使用cumprod（）来表示
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)  # [N_rays,

        # 计算每个采样点沿着每个射线的加权颜色。
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # 估计的深度图是预期距离。
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # 视差图是深度的倒数。
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                tf.reduce_sum(weights, axis=-1))

        # 沿着每条射线的权重总和。这个值在数值误差上是在[0，1]范围内的。
        acc_map = tf.reduce_sum(weights, -1)

        # 如果要合成到白色背景上，则使用累积的alpha图。
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # 获取批次大小
    N_rays = ray_batch.shape[0]

    # 提取射线起点和方向
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # 提取单位归一化的视线方向
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # 提取射线距离的下限和上限
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # 决定沿每条射线采样的位置。根据逻辑，所有射线将在相同的时间采样。
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # 在 'near' 和 'far' 之间线性地进行空间积分。所有射线将使用相同的积分点。
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # 在逆深度（视差）中线性采样。
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # 扰动每条射线的采样时间。
    if perturb > 0.:
        # 获取采样间隔
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # 在这些间隔中进行分层采样
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # 评估模型的空间点。
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # 在每个点上评估模型。
    raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        # 记录下粗糙模型预测的rgb_map，disp_map，acc_map
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # 基于粗略模型中为每个颜色分配的权重，获取额外的积分时间
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # 获取用于评估颜色和密度的所有点。
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # 使用network_fine进行预测。
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        # 获取输出
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    # 如果retraw为真，将原始值 raw 添加到字典 ret 中
    if retraw:
        ret['raw'] = raw
    # 如果 N_importance 大于 0，则将 rgb_map_0、disp_map_0、acc_map_0 和 z_samples 标准差添加到字典 ret 中
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1) # [N_rays]
    # 对字典 ret 中的每个键值对进行数值检查
    for k in ret:
        tf.debugging.check_numerics(ret[k], '输出 {}'.format(k))
    # 返回字典 ret
    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """将射线分批渲染以避免OOM。
    @param rays_flat: 一维形状为[N, ...]的张量，包含N个射线。
    @param chunk: minibatch的大小，一个标量。
    @param **kwargs: 用于render_rays函数的可选参数。
    @return all_ret: 包含render_rays函数所有返回值的字典。
    """
    all_ret = {}
    # 按块大小chunk循环，渲染小批量射线
    for i in range(0, rays_flat.shape[0], chunk):
        # 渲染当前批量射线，并将所有返回值存储到ret字典中
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        # 将ret字典中的每个键值对添加到all_ret字典中
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # 将all_ret字典中的每个值连接成张量
    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """光线渲染
    @param H: int。图像的高度（像素）。
    @param W: int。图像的宽度（像素）。
    @param focal: float。针孔相机的焦距。
    @param chunk: int。同时处理的最大射线数。用于控制最大内存使用量。不影响最终结果。
    @param rays: 形状为[2, batch_size, 3]的数组。每个示例的射线起点和方向。
    @param c2w: 形状为[3, 4]的数组。相机到世界的变换矩阵。
    @param ndc: bool。如果为True，则表示射线起点和方向在NDC坐标中。
    @param near: float或形状为[batch_size]的数组。射线的最近距离。
    @param far: float或形状为[batch_size]的数组。射线的最远距离。
    @param use_viewdirs: bool。如果为True，则使用模型中一个点的观察方向。
    @param c2w_staticcam: 形状为[3, 4]的数组。如果不为None，则使用此变换矩阵作为相机，同时使用其他c2w参数作为观看方向。

    @return rgb_map: [batch_size, 3]。射线的预测RGB值。
    @return disp_map: [batch_size]。视差图。深度的倒数。
    @return acc_map: [batch_size]。沿射线累积的不透明度（alpha）。
    @return extras: 包含render_rays()返回值的字典。
    """

    if c2w is not None:
        # 特殊情况：渲染整个图像
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # 使用提供的射线批量
        rays_o, rays_d = rays

    if use_viewdirs:
        # 将射线方向作为输入
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # 特殊情况：使用viewdirs的效果可视化
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # 将所有方向变为单位矢量。
        # 形状：[batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # 对于前向场景
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # 创建射线批量
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # 每条射线的(射线起点，射线方向，最小距离，最大距离)
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # 渲染并重塑
    # 将射线分成块并进行批处理
    all_ret = batchify_rays(rays, chunk, **kwargs)

    # 对于每个返回值，调整其形状以匹配其他返回值的形状
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    # 从所有返回值中提取rgb_map、disp_map和acc_map，将它们存储在一个列表中
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]

    # 将所有不在k_extract列表中的返回值存储在一个字典中
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    # 返回ret_list和ret_dict列表
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """渲染路径中的每个视角并返回结果
    @param render_poses: 渲染视角的相机姿态列表
    @param hwf: 相机的高度、宽度和焦距
    @param chunk: 渲染射线的大小
    @param render_kwargs: 渲染器参数的字典
    @param gt_imgs: 可选，与渲染图像进行比较的参考图像列表
    @param savedir: 可选，保存渲染图像的目录路径
    @param render_factor: 可选，用于加速渲染的降采样因子
    @return rgbs: 渲染的RGB图像列表
    @return disps: 渲染的深度图像列表
    """

    # 获取相机的高度、宽度和焦距
    H, W, focal = hwf

    if render_factor != 0:
        # 如果指定了渲染因子，则将高度、宽度和焦距缩小以加快渲染
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    # 初始化空的RGB和深度图像列表
    rgbs = []
    disps = []

    t = time.time()
    # 遍历所有渲染视角的相机姿态
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        # 渲染RGB、深度和权重图像
        rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        # 将渲染的RGB和深度图像添加到列表中
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        # 如果提供了参考图像，则计算并打印PSNR
        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        # 如果提供了保存目录，则将RGB图像保存为PNG文件
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    # 将所有渲染的RGB和深度图像堆叠在一起
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """实例化 NeRF MLP 模型。
    @param args: argparse.ArgumentParser 解析出的命令行参数
    @return render_kwargs_train: 渲染器训练参数字典
    @return render_kwargs_test: 渲染器测试参数字典
    @return start: 训练起始步骤
    @return grad_vars: 可训练变量的列表
    @return models: 包含 'model' 和 'model_fine' 的字典，其中 'model' 是主要的 MLP 模型，'model_fine' 用于重要性采样。
    """

    # 获取嵌入函数和输入通道数
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # 如果需要视角编码，则获取嵌入函数和输入通道数
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    # 初始化 MLP 模型
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        # 如果使用重要性采样，则初始化额外的 MLP 模型
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    # 定义网络查询函数，输入和视角查询嵌入，并调用运行网络
    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    # 训练和测试渲染器参数
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # 如果数据集类型不是 LLFF，则不使用 NDC
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # 测试时不进行扰动
    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    # 如果有 fine-tune 模型，则使用 fine-tune 模型，否则使用最新的训练模型
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        # 加载最新的训练模型
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            # 如果有 fine-tune 模型，则使用 fine-tune 模型，否则使用最新的训练模型
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    
    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    # 解析命令行参数并返回一个argparse.ArgumentParser 对象的实例
    parser = config_parser() 
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果设置了随机数种子参数，就使用它来设置随机种子
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed) # numpy种子
        tf.compat.v1.set_random_seed(args.random_seed) # tensorflow种子

    # Load data 加载数据
    if args.dataset_type == 'llff':
        # 加载 LLFF 数据集
        # recenter: 是否将图像中心移动到坐标原点
        # bd_factor: 相机远近平面与包围盒之间的距离因子
        # spherify: 是否将场景球形化（将相机半径设置为平均相机距离）
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=args.spherify)
        hwf = poses[0, :3, -1]  # 相机的高度、宽度、焦距
        poses = poses[:, :3, :4]  # 前 3 行为相机的旋转矩阵，最后一列为相机的平移向量
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        # 如果设置了自动 LLFF holdout 参数，则自动划分测试集
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        # 训练集为非测试集的所有图像
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        # 定义边界，如果没有设置 ndc 参数，则使用最小和最大深度值来定义边界
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    # 加载 Blender 数据集
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2. # 设置边界
        far = 6.

        if args.white_bkgd:
            # 如果设置了白色背景参数，则将 RGB 值与 alpha 值相乘并添加 alpha 值的补数来实现白色背景
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':
        # 加载 DeepVoxels 数据集
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        # 设置边界
        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    # 将相机内参类型转换为正确类型
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 如果需要渲染测试图像，则使用测试集的视角参数进行渲染
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建日志目录并复制配置文件
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 创建 Nerf 模型并设置渲染参数
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)

    # 设置近和远平面距离并更新渲染参数
    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # 如果只是渲染模型的输出，则进行短路处理
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test 会切换到测试姿态
            images = images[i_test]
        else:
            # 默认是平滑的render_poses路径
            images = None

        # 创建保存目录
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        # 渲染路径
        rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        # 将图像保存为视频
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=8)

        return

    # 创建优化器
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    # 获取全局步数
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # 如果批处理随机光线，则准备光线批处理张量
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # 对于随机光线批处理。
        # 
        # 构造形状为[N*H*W, 3, 3]的rays_rgb数组，
        # 其中axis=1被解释为，
        #   axis=0：世界空间中的光线起点
        #   axis=1：世界空间中的光线方向
        #   axis=2：像素的观察到的RGB颜色
        print('get rays')
        # get_rays_np()返回rays_origin=[H, W, 3], rays_direction=[H, W, 3]的二元组
        # 为图像中的每个像素。stack()添加一个新维度。
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # [N_train, H, W, ro+rd+rgb, 3]，只用训练图像
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)
        # [(N-1)*H*W, ro+rd+rgb, 3]，展平数组以便在随机采样的时候可以方便地批处理
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch

        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:

            rgbs, disps = render_path(
                render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                rgbs_still, _ = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0 or i < 10:

            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)

            if i % args.i_img == 0:

                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3, :4]

                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))
                
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i==0:
                    os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    train()
