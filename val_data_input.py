import csv  
import re
import os

def val_data_input(subject='1',story='all',video_root='./face_val',audio_root='./audio_train'):

	video_dirs=os.listdir(video_root)

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
			
			data=[]

			video_dir=os.path.join(video_root,dir_name)
			audio_dir=os.path.join(audio_root,dir_name)

			frames=os.listdir(video_dir)

			for frame in frames:
				frame_data=[]
				order=re.findall(r'\d+',frame)[0]
				frame_data.append(order)
				frame_data.append(os.path.join(video_dir,frame))
				frame_data.append(os.path.join(audio_dir,order+'.txt'))
				frame_data.append(float(order)/len(frames))
				data.append(frame_data)

			yield dir_name,data

if __name__ == '__main__':

	datas = val_data_input()

	for data in datas:
		print(data[0])
		input()

