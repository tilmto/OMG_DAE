import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from train_data_input import *
from val_data_input import *
from prediction_complement import *

class DAE:

	def __init__(self,images,audios,labels,times,keep_prob=1):
		self.images=images
		self.audios=audios
		self.labels=labels
		self.times=times
		self.batch_size=tf.cast(images.shape[0],tf.int32)
		self.keep_prob=keep_prob

	def build_model(self):

		self.shared_feature_extract(300,100)

		self.build_empathy_model()

		w=0

		l2_loss=tf.add_n(tf.get_collection('weight'))
		self.empathy_cost=tf.reduce_sum(tf.square(self.empathy-self.labels))+w*l2_loss
		self.empathy_train_op=tf.train.AdamOptimizer(1e-4).minimize(self.empathy_cost)

	def shared_feature_extract(self,video_dim,audio_dim):
		video_part=self.video_encoder(video_dim)
		audio_part=self.audio_encoder(audio_dim)
		shared_feature_pre=tf.concat([video_part,audio_part],1)

		self.shared_feature=self.fully_connected(shared_feature_pre,video_dim+audio_dim)

	def build_empathy_model(self):

		hidden_dim1=200
		hidden_dim2=50

		t=tf.nn.relu(self.fully_connected(self.times,50))
		t=tf.nn.dropout(t,self.keep_prob)
		t=self.fully_connected(t,hidden_dim1)

		x=self.shared_feature
		x=self.fully_connected(x,hidden_dim1)

		w=flags.w_distrib
		x=tf.nn.relu(w*t+(1-w)*x)
		x=tf.nn.dropout(x,self.keep_prob)

		x=tf.nn.relu(self.fully_connected(x,hidden_dim2))
		x=tf.nn.dropout(x,self.keep_prob)

		self.empathy=tf.nn.tanh(self.fully_connected(x,1))

	def video_encoder(self,video_dim):

		filters=[16,64,128,256]
		strides=[1,2,2]
		num_unit=[1,1,1]

		x=self.images
		x=tf.nn.relu(self.conv2d(x,3,3,filters[0],1))
		x=self.max_pool(x)

		x=self.bottleneck_residual(x,filters[0],filters[1],strides[0])
		for i in range(num_unit[0]):
			x=self.bottleneck_residual(x,filters[1],filters[1],1)

		x=self.bottleneck_residual(x,filters[1],filters[2],strides[1])
		for i in range(num_unit[1]):
			x=self.bottleneck_residual(x,filters[2],filters[2],1)

		x=self.bottleneck_residual(x,filters[2],filters[3],strides[2])
		for i in range(num_unit[2]):
			x=self.bottleneck_residual(x,filters[3],filters[3],1)

		x=self.global_avg_pool(x)
		x=self.fully_connected(x,video_dim)

		return x

	def bottleneck_residual(self,x,in_filter,out_filter,stride):
		orig_x=x
		x=tf.nn.relu(self.conv2d(x,1,in_filter,int(out_filter/4),stride))
		x=tf.nn.relu(self.conv2d(x,3,int(out_filter/4),int(out_filter/4),1))
		x=self.conv2d(x,1,int(out_filter/4),out_filter,1)

		if(in_filter!=out_filter):
			orig_x=self.conv2d(orig_x,1,in_filter,out_filter,stride)
		x=tf.nn.relu(x+orig_x)
		return x

	def audio_encoder(self,audio_dim):
		hidden_dim=300
		x=self.audios

		x=tf.nn.relu(self.fully_connected(x,hidden_dim))
		x=tf.nn.dropout(x,self.keep_prob)
		
		x=tf.nn.relu(self.fully_connected(x,audio_dim))
		return x

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape,stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.1,shape=shape)
		return tf.Variable(initial)

	def conv2d(self,x,filter_size,in_filter,out_filter,stride):
		W=self.weight_variable([filter_size,filter_size,in_filter,out_filter])
		tf.add_to_collection('weight',tf.nn.l2_loss(W))
		return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')

	def conv2d_transpose(self,x,filter_size,in_filter,out_filter,img_size):
		W=self.weight_variable([filter_size,filter_size,out_filter,in_filter])
		tf.add_to_collection('weight',tf.nn.l2_loss(W))
		return tf.nn.conv2d_transpose(x,W,output_shape=[self.batch_size,img_size,img_size,out_filter],strides=[1,2,2,1],padding='SAME')

	def max_pool(self,x):
		return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	def global_avg_pool(self, x):
		return tf.reduce_mean(x, [1, 2])

	def fully_connected(self,x,out_dim):
		x=tf.reshape(x,[self.batch_size,-1])
		W=self.weight_variable([tf.cast(x.shape[1],tf.int32),out_dim])
		b=self.bias_variable([out_dim])
		tf.add_to_collection('weight',tf.nn.l2_loss(W))
		tf.add_to_collection('weight',tf.nn.l2_loss(b))
		return tf.matmul(x,W)+b

	def get_data(self,images,audios,labels,times,keep_prob=1):
		self.images = images
		self.audios = audios
		self.labels = labels
		self.times = times
		self.batch_size = tf.cast(images.shape[0],tf.int32)
		self.keep_prob = keep_prob



tf.app.flags.DEFINE_string('subject', '1', 'Specify the subject 1-10')

tf.app.flags.DEFINE_string('story', 'all', 'Specify the story 2,4,5,8')

tf.app.flags.DEFINE_string('saver', 'general', 'Specify the saver')

tf.app.flags.DEFINE_string('video_train', './face_train', 'Specify the path for video_train')

tf.app.flags.DEFINE_string('audio_train', './audio_train', 'Specify the path for audio_train')

tf.app.flags.DEFINE_string('video_val', './face_val', 'Specify the path for video_val')

tf.app.flags.DEFINE_string('audio_val', './audio_train', 'Specify the path for audio_val')

tf.app.flags.DEFINE_boolean('train_mode', True, 'Whether train mode or not')

tf.app.flags.DEFINE_boolean('no_zero', False, 'Whether no sample with label 0 or not')

tf.app.flags.DEFINE_boolean('go_on_last_train', False, 'Whether going on training from last stop step')

tf.app.flags.DEFINE_integer('batch_size', 16, 'Input the batch size')

tf.app.flags.DEFINE_integer('steps', 10000, 'Input the training steps')

tf.app.flags.DEFINE_float('w_distrib', 1.0, 'Input the weight of distribution learning')

flags = tf.app.flags.FLAGS

def main(_):

	train_mode = True

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	if flags.saver == 'general':
		saver_path = './ckpt/ckpt_'+flags.subject
	else:
		saver_path = flags.saver

	with tf.Session(config=config) as sess:
		if flags.train_mode:

			images_train,audios_train,labels_train, times_train = train_data_input(batch_size=flags.batch_size,subject=flags.subject,story=flags.story,no_zero=flags.no_zero,video_root=flags.video_train,audio_root=flags.audio_train)
			dae = DAE(images_train,audios_train,labels_train,times_train,0.7)
			dae.build_model()

			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			saver = tf.train.Saver(max_to_keep=10)

			coord = tf.train.Coordinator() 
			threads = tf.train.start_queue_runners(sess=sess,coord=coord)

			if flags.go_on_last_train:
				saver_res = tf.train.Saver()
				model_file=tf.train.latest_checkpoint(saver_path)
				saver_res.restore(sess,model_file)

			print('Training:')
			for i in range(1,flags.steps+1):
				try:
					_, empathy_cost = sess.run([dae.empathy_train_op,dae.empathy_cost])
					print('step:',i,' empathy_cost:', empathy_cost)

					if not i%10000:
						saver.save(sess, saver_path+'/dae.ckpt', global_step = i)
					
					print('***********************************')

				except OutOfRangeError:
					images_train,audios_train,labels_train, times_train = train_data_input(batch_size=flags.batch_size,subject=flags.subject,story=flags.story,no_zero=flags.no_zero,video_root=flags.video_train,audio_root=flags.audio_train)
					dae.get_data(images_train,audios_train,labels_train,times_train,0.7)

			saver.save(sess, saver_path+'/dae.ckpt', global_step = i)

			coord.request_stop()
			coord.join(threads)

		else:

			image_placeholder = tf.placeholder(tf.float32,shape = [1,120,120,3])
			audio_placeholder = tf.placeholder(tf.float32,shape = [1,640])
			times_placeholder = tf.placeholder(tf.float32,shape = [1,1])

			dae = DAE(image_placeholder,audio_placeholder,0,times_placeholder,1)
			dae.build_model()

			saver = tf.train.Saver()
			model_file=tf.train.latest_checkpoint(saver_path)
			saver.restore(sess,model_file)

			val_datas = val_data_input(subject=flags.subject, story=flags.story, video_root=flags.video_val, audio_root=flags.audio_val)

			for val_data in val_datas:
				with open('./val_predict_raw/'+val_data[0]+'.csv', 'w', newline='') as f:
					writer = csv.writer(f) 
					writer.writerow(['order','valence'])

					print('Dealing with ' + val_data[0])

					empathy_list = []

					for i in range(len(val_data[1])):

						image_val = Image.open(val_data[1][i][1])
						image_val = image_val.convert('RGB')
						image_val = image_val.resize((120,120))
						image_val = np.array(image_val)
						image_val = image_val / 255
						image_val = np.reshape(image_val,[1,120,120,3])

						audio_val = np.loadtxt(val_data[1][i][2],dtype=np.float32)
						audio_val = np.reshape(audio_val,[1,640])

						time_val = np.reshape(val_data[1][i][3],[1,1])

						empathy = dae.empathy.eval(feed_dict = {image_placeholder: image_val, audio_placeholder:audio_val, times_placeholder:time_val})

						empathy_list.append([int(val_data[1][i][0]),empathy[0][0]])

					empathy_list.sort(key=lambda x:x[0])

					writer.writerows(empathy_list)

					print('Complete ' + val_data[0])
					print('***********************************')

			prediction_complement()

 
if __name__ == '__main__':
	tf.app.run()
		
