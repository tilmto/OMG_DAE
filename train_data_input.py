import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import csv
import re

def train_data_input(batch_size=20, subject='1', story='all', no_zero=False, video_root='./face_train', audio_root='./audio_train', label_path='./Annotations'):

	frame_path=[]
	audio_path=[]
	labels=[]
	time_list=[]

	video_dirs=os.listdir(video_root)

	print('Preparing data:')
	for dir_name in video_dirs:
		if dir_name.find('Subject_'+subject+'_')!=-1:

			if story!='all':
				story_list=story.split(',')

				flag=True
				for s in story_list:
					if dir_name.find('Story_'+s)!=-1:
						flag=False
						break
						
				if flag:
					continue

			with open(os.path.join(label_path,dir_name+'.csv')) as f:
				reader=csv.reader(f)
				head_row=next(reader)
				labels_orig=[]
				for row in reader:
					labels_orig.append(row[0])

			audio_dir=os.path.join(audio_root,dir_name)

			video_dir=os.path.join(video_root,dir_name)
			frames=os.listdir(video_dir)

			for frame in frames:
				time=re.findall(r'\d+',frame)[0]
				label=float(labels_orig[int(time)-1])
				if label==0 and no_zero:
					continue
				labels.append(label)
				audio_path.append(os.path.join(audio_dir,time+'.txt'))
				frame_path.append(os.path.join(video_dir,frame))
				time_list.append(float(time)/len(frames))

			print(dir_name)

	input_queue = tf.train.slice_input_producer([frame_path, audio_path, labels, time_list], num_epochs=None, shuffle=True)

	image = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image, channels = 3)	
	image = tf.image.resize_images(image, (120,120), np.random.randint(4))
	image = image / 255

	audio_file = tf.read_file(input_queue[1])
	audio_indice = tf.string_split([audio_file],delimiter='\r\n')
	audio = tf.string_to_number(audio_indice.values)
	audio = tf.reshape(audio,[640])

	label = [input_queue[2]]

	time = [input_queue[3]]

	image_batch, audio_batch, label_batch , time_batch = tf.train.shuffle_batch([image, audio, label, time], batch_size = batch_size, capacity = batch_size*200, min_after_dequeue = batch_size*20, num_threads = 32)
	 
	return image_batch, audio_batch, label_batch, time_batch


if __name__ == '__main__':
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		image_batch, audio_batch, label_batch, time_batch = train_data_input(batch_size=3,subject='5',story='2')

		coord = tf.train.Coordinator() 
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)

		for i in range(5):
			images, audios, labels, times = sess.run([image_batch, audio_batch, label_batch, time_batch])
			print(images, audios, labels, times)
			input()

		coord.request_stop()
		coord.join(threads)
