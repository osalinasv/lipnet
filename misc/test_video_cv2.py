import cv2
import os


VIDEO_CODEC = 'MP4V'
VIDEO_EXT   = '.mp4'
FRAME_RATE  = 25

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_COLOR = (0, 255, 255)
FONT_SCALE = 0.75
FONT_THICK = 1


def split_text_per_frame(text: str, video_frames_len: int) -> [str]:
	subs = text.split()
	inc  = max(video_frames_len / (len(subs) + 1), 0.01)

	return [" ".join(subs[:int(i / inc)]) for i in range(0, video_frames_len)]


def get_text_size(text: str) -> (int, int):
	return cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)[0]


def get_text_position(image_size: (int, int), text_size: (int, int), bottom_offset: int) -> (int, int):
	x = (image_size[0] - text_size[0]) // 2
	y = image_size[1] - text_size[1] - bottom_offset

	return x, y


def put_text(image, text: str, text_position: (int, int)):
	cv2.putText(image, text, text_position, FONT, FONT_SCALE, FONT_COLOR, FONT_THICK)


def get_image_size(capture) -> (int, int):
	return int(capture.get(3)), int(capture.get(4))


splits = split_text_per_frame('set blue with e six again', 75)
output_path = os.path.realpath(os.path.join(os.path.dirname(__file__), 'video' + VIDEO_EXT))

capture = cv2.VideoCapture('D:/GRID/s34/sbwe6a.mpg')

if capture.isOpened():
	image_size = get_image_size(capture)

	fourcc     = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
	out_writer = cv2.VideoWriter(output_path, fourcc, FRAME_RATE, image_size)

	frame_count = 0

	while capture.isOpened():
		ret, frame = capture.read()

		if ret == True:
			if 0 < frame_count <= len(splits):
				text = splits[frame_count]

				text_size = get_text_size(text)
				text_pos  = get_text_position(image_size, text_size, 30)

				put_text(frame, text, text_pos)

			out_writer.write(frame)
			frame_count += 1
		else:
			break

	out_writer.release()

capture.release()
