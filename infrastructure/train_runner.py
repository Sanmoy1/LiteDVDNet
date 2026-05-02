import importlib
import os
import shutil
import time
import torch

import random
import numpy as np
import torch.nn.functional as F_ops  # for SR ground truth upsampling
from torch import nn, optim
from dataloaders import train_dali_loader
from dataset import ValDataset

from train_common import resume_training, save_model_checkpoint, need_ortog, binary_lr_scheduler
from utils import normalize_augment, init_logging, svd_orthogonalization, close_logger, load_options, batch_psnr
from degradation import apply_complex_degradation
from skimage.metrics import structural_similarity as ssim


class TrainRunner:
	def __init__(self, options_path: str):
		self.options_path = options_path
		self.options = self.parse_options(options_path) #dictionary for noise parameters

	def parse_options(self, options_path) -> dict:

		opt = load_options(options_path)

		# Normalize noise between [0, 1]
		opt['val_noiseL'] /= 255.
		opt['noise_ival'][0] /= 255.
		opt['noise_ival'][1] /= 255.

		return opt

	def get_options(self) -> dict:
		return self.options

	def get_model(self) -> nn.Module:
		opts = self.get_options()

		module = importlib.import_module(opts['module'])
		model_class = getattr(module, opts['model_name'])
		instance = model_class(**opts['model_params'])

		return instance

	def set_seed(self, args): #for similar initial weights for reproducibility
		seed = args['manual_seed']

		if seed is None:
			seed = random.randint(1, 10000)
			args['manual_seed'] = seed

		# set seeds for all random sources
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	def validate_training_options(self, args):

		# Validate params before training starts
		milestones = args['milestones']
		learning_rates = args['learning_rates']
		orthog_epochs = args['orthog_epochs']
		epochs = args['epochs']

		if (len(learning_rates) != len(milestones)):
			raise ValueError("Milestones and learning rates must match!")

		if not all(item <= epochs for item in milestones):
			raise ValueError("Milestones should not exceed training epochs!")

		if (orthog_epochs >= epochs):
			raise ValueError("Orthog epochs should not exceed training epochs!")

	def train(self):
			r"""Performs the main training loop
			"""
			args = self.get_options()

			# self.validate_training_options(args)
			self.set_seed(args)

			# Load dataset
			print('> Loading datasets ...')
			valset_dir = args.get('valset_dir', '')
			if valset_dir and os.path.isdir(valset_dir):
				dataset_val = ValDataset(valsetdir=valset_dir, gray_mode=False)
			else:
				dataset_val = []
				print('> Validation set not found or not specified. Skipping validation.')
			loader_train = train_dali_loader(batch_size=args['batch_size'], \
											 file_root=args['trainset_dir'], \
											 sequence_length=args['temp_patch_size'], \
											 crop_size=args['patch_size'], \
											 epoch_size=args['max_number_patches'], \
											 random_shuffle=True, \
											 temp_stride=3)

			# Define GPU devices
			torch.backends.cudnn.benchmark = True  # CUDNN optimization

			# Create model
			model = self.get_model()
			args['model_description'] = model.get_desciption()
			model = nn.DataParallel(model, device_ids=[0]).cuda()
			args['log_dir'] = os.path.join(args.get('experiments_dir', './experiments'), args['model_description'])
			isResumedTraining = args['resume_training']

			if os.path.exists(args['log_dir']) and isResumedTraining is False:
				# Archive the existing folder instead of crashing — renames it to {name}_archived_{timestamp}
				import time as _time
				archive_name = args['log_dir'] + '_archived_' + str(int(_time.time()))
				os.rename(args['log_dir'], archive_name)
				print(f"> Existing experiment folder archived to: {archive_name}")


			# Init loggers
			writer, logger = init_logging(args)


			# Show training characteristics
			num_minibatches = int(args['max_number_patches'] // args['batch_size'])
			ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
			logger.info(f'Training power: {args["epochs"] * num_minibatches} minibatches')


			# Copy options
			options_copy_path = os.path.join(args['log_dir'], f"{args['model_description']}.yaml")
			shutil.copy(self.options_path, options_copy_path)

			# Define loss
			criterion = nn.L1Loss(reduction='sum')
			criterion.cuda()

			# Optimizer
			optimizer = optim.Adam(model.parameters(), lr=args['lr'])

			# Resume training or start anew
			start_epoch, training_params = resume_training(args, model, optimizer)

			# Training
			start_time = time.time()
			for epoch in range(start_epoch, args['epochs'] + 1):

				# Set learning rate
				current_lr = binary_lr_scheduler(epoch, args)

				# Set learning rate in optimizer
				for param_group in optimizer.param_groups:
					param_group["lr"] = current_lr

				training_params['orthog_enabled'] = need_ortog(epoch, args)

				print(f'\n[epoch {epoch}]Learning rate: {current_lr}')
				print(f"[epoch {epoch}]Orthogonalization: {training_params['orthog_enabled']}")

				epoch_start_time = time.time()

				# Train
				for i, data in enumerate(loader_train, 0):

					# Pre-training step
					model.train()

					# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
					optimizer.zero_grad()

					# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
					# extract ground truth (central frame)
					img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx)
					N, _, H, W = img_train.size()

					# Send clean frames to GPU first (degradation runs on GPU)
					img_train = img_train.cuda(non_blocking=True)
					gt_train = gt_train.cuda(non_blocking=True)

					# Apply complex degradation pipeline (Gaussian + Poisson + Blur + JPEG)
					# stdn is the Gaussian noise std used as the noise_map estimate for the model
					imgn_train, stdn = apply_complex_degradation(img_train, args['degradation'])

					#imgn_train -> LQ

					noise_map = stdn.expand((N, 1, H, W))  # one channel per image, already on GPU

					# Evaluate model and optimize it (model output)
					out_train = model(imgn_train, noise_map)

					# Compute loss
					# If SR is enabled, model outputs HR — upsample LR ground truth to match.
					# When use_sr=False (or scale=1), gt_for_loss == gt_train (backward compatible).
					sr_scale = args['model_params'].get('scale', 1)
					if args['model_params'].get('use_sr', False) and sr_scale > 1:
						gt_for_loss = F_ops.interpolate(gt_train, scale_factor=sr_scale,
						                                mode='bicubic', align_corners=False).clamp(0., 1.)
					else:
						gt_for_loss = gt_train #gt_for_loss -> GT (no Super Resolution)
					loss = criterion(gt_for_loss, out_train) / (N * 2)
					loss.backward()
					optimizer.step()

					# Results
					if training_params['step'] % args['save_every'] == 0:
						# Apply regularization by orthogonalizing filters
						if training_params['orthog_enabled']:
							model.apply(svd_orthogonalization)

						print(f"[epoch {epoch}][{i + 1}/{num_minibatches}] Loss: {loss.item():1.4f}")

					# update step counter
					training_params['step'] += 1


				# Call to model.eval() to correctly set the BN layers before inference
				model.eval()

				# Validation
				if len(dataset_val) > 0:
					psnr_val, ssim_val = self.calculate_metrics(model, dataset_val, args['val_noiseL'], args['temp_patch_size'])
					# Log validation results
					print(f"[epoch {epoch}] PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
					logger.info(f"[epoch {epoch}] PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
				else:
					print(f"[epoch {epoch}] Validation skipped (no validation data found)")


				# save model and checkpoint
				training_params['start_epoch'] = epoch
				save_model_checkpoint(model, args, optimizer, training_params, epoch)

				# estimate time left
				epoch_time = time.time() - epoch_start_time
				time_left = (args['epochs'] - epoch) * epoch_time
				print(f'Epoch {epoch} training time: {self.format_time(epoch_time)}. '
					  f'Time left: {self.format_time(time_left)}.')

			# Print elapsed time
			elapsed_time = time.time() - start_time
			print(f'Elapsed time {self.format_time(elapsed_time)}')

			self.save_results(args['model_description'], args['log_dir'], logger)

			# Close logger file
			close_logger(logger)

	def save_results(self, model_desc, log_dir, logger):

		save_dir = os.path.join('models', model_desc)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		else:
			logger.info('{} already exists!'.format(save_dir))
			return

		# Copy options
		src_path = os.path.join(log_dir, f'{model_desc}.yaml')
		dst_path = os.path.join(save_dir, f'{model_desc}.yaml')
		shutil.copy(src_path, dst_path)

		# Copy model
		src_path = os.path.join(log_dir, f'{model_desc}.pth')
		dst_path = os.path.join(save_dir, f'{model_desc}.pth')
		shutil.copy(src_path, dst_path)

		# Copy logs
		src_path = os.path.join(log_dir, 'log.txt')
		dst_path = os.path.join(save_dir, 'log.txt')
		shutil.copy(src_path, dst_path)

	def format_time(self, time_to_format):
		return time.strftime("%H:%M:%S", time.gmtime(time_to_format))

	def calculate_metrics(self, model, dataset_val, valnoisestd, temp_psz):
		"""Single-pass validation: computes PSNR and SSIM together to avoid running
		the model twice. Returns (psnr_val, ssim_val) averaged over all sequences."""
		from skimage.metrics import structural_similarity as ssim_fn

		psnr_val   = 0.0
		ssim_val   = 0.0
		total_frames = 0

		with torch.no_grad():
			for seq_val in dataset_val:
				# --- add noise ---
				noise    = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
				seqn_val = (seq_val + noise).cuda()

				sigma_noise = torch.tensor([valnoisestd], dtype=torch.float32, device='cuda')
				numframes, C, H, W = seqn_val.shape
				noise_map = sigma_noise.expand((1, 1, H, W))

				# --- run model once ---
				out_val = self.denoise_seq(model=model, seq=seqn_val,
				                          noise_map=noise_map, temp_psz=temp_psz)

				# --- build clean reference (SR-aware) ---
				# seq_val is [numframes, C, H, W]; keep it that way for interpolate
				clean_ref = seq_val.squeeze_()
				if out_val.shape[-2:] != clean_ref.shape[-2:]:
					if clean_ref.dim() == 3:   # single frame [C, H, W]
						clean_ref = F_ops.interpolate(clean_ref.unsqueeze(0),
						                              size=out_val.shape[-2:],
						                              mode='bicubic',
						                              align_corners=False).squeeze(0)
					else:                       # multi-frame [numframes, C, H, W]
						clean_ref = F_ops.interpolate(clean_ref,
						                              size=out_val.shape[-2:],
						                              mode='bicubic',
						                              align_corners=False)

				# --- PSNR (sequence-level, then averaged over sequences) ---
				psnr_val += batch_psnr(out_val.cpu(), clean_ref, 1.)

				# --- SSIM (frame-level accumulation) ---
				out_np   = out_val.cpu().numpy()
				clean_np = clean_ref.cpu().numpy()
				for fi in range(out_np.shape[0]):
					ssim_val += ssim_fn(
						clean_np[fi].transpose(1, 2, 0),   # [H, W, C]
						out_np[fi].transpose(1, 2, 0),
						data_range=1.0,
						channel_axis=2
					)

				total_frames += numframes

		# PSNR: average over sequences; SSIM: average over all frames
		psnr_val /= len(dataset_val)
		ssim_val  /= total_frames

		return psnr_val, ssim_val


	def denoise_seq(self, model, seq, noise_map, temp_psz):
		r"""Denoises a sequence of frames with FastDVDnet.

		Args:
			seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
			noise_map: Tensor. Noise map
			temp_psz: size of the temporal patch
			model_temp: instance of the PyTorch model of the temporal denoiser
		Returns:
			denframes: Tensor, [numframes, C, H, W]
		"""
		# init arrays to handle contiguous frames and related patches
		numframes, C, H, W = seq.shape
		ctrlfr_idx = int((temp_psz-1)//2)
		inframes = list()
		denframes = None   # allocated lazily from first output (handles SR size automatically)

		for fridx in range(numframes):
			# load input frames
			if not inframes:
			# if list not yet created, fill it with temp_patchsz frames
				for idx in range(temp_psz):
					relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
					inframes.append(seq[relidx])
			else:
				del inframes[0]
				relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
				inframes.append(seq[relidx])

			inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)

			# run model
			out_frame = torch.clamp(model(inframes_t, noise_map), 0., 1.)  # [1, C, H_out, W_out]

			# allocate denframes on first iteration using actual output shape (SR-safe)
			if denframes is None:
				_, C_out, H_out, W_out = out_frame.shape
				denframes = torch.empty((numframes, C_out, H_out, W_out)).to(seq.device)

			denframes[fridx] = out_frame.squeeze(0)

		# free memory up
		del inframes
		del inframes_t
		torch.cuda.empty_cache()

		# convert to appropiate type and return
		return denframes





