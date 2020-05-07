import os
import glob
import argparse
import matplotlib
from PIL import Image
from zipfile import ZipFile, ZIP_STORED

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_each_depth_map
from matplotlib import pyplot as plt

def get_model(wt='nyu.h5'):
	# Custom object needed for inference and training
	custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
	print('Loading model...')
	# Load model into GPU / CPU
	model = load_model(wt, custom_objects=custom_objects, compile=False)
	print('\nModel loaded ({0}).'.format(wt))
	return model

def infer(model, bg_idx, base_path="/content/fg_bg", show_viz=False, zipf=None,
		batch_size=128, scale=0):
	path = f"{base_path}/bg_{bg_idx:02d}/*.jpg"
	print('\nLoading images from %s' % path)
	# Input images
	input_files = glob.glob(path)
	output_files = list(map(lambda f: f.split("/")[-1][:-4]+".jpg", input_files))

	batch_idx = 0
	for start in range(0, len(input_files), batch_size):
		batch_idx += 1
		end = min(len(input_files), start + batch_size)

		inputs = load_images( input_files[start:end], scale=scale )
		print('Batch[{0}]: Loaded ({1}) images of size {2}.'.format(batch_idx,
			inputs.shape[0], inputs.shape[1:]))
		# Compute results
		outputs = predict(model, inputs)

		new_zip = False
		if not zipf:
			new_zip = True
			zip_file_name = "fg_bg_depth.zip"
			print("Creating archive: {:s}".format(zip_file_name))
			zipf = ZipFile(zip_file_name, mode='a', compression=ZIP_STORED)

		# Save results
		out_fnames = output_files[start:end]
		assert(outputs.shape[0] == len(out_fnames))
		save_each_depth_map(outputs.copy(), out_fnames, zipf=zipf,
							out_prefix=f"bg_{bg_idx:02d}/")

		if new_zip:
			zipf.close()
		
		if show_viz and batch_idx==1:
			# Display results
			viz = display_images(outputs[:16].copy(), inputs[:16].copy(), cmap="gray")
			plt.figure(figsize=(10,5))
			plt.axis("off")
			plt.imshow(viz)
			plt.savefig('visualise_depth_maps.jpg')
			plt.close()

if __name__ == '__main__':
	model = get_model()
	zip_file_name = "fg_bg_depth.zip"
	print("Creating archive: {:s}".format(zip_file_name))
	zipf = ZipFile(zip_file_name, mode='a', compression=ZIP_STORED)
	infer(model, 0, zipf=zipf, batch_size=512, scale=0)
	zipf.close()