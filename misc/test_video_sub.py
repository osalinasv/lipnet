# TODO: This script depends on moviepy and ImageMagick to work. However I simply could not make moviepy recognize the
# install location of ImageMagick on WIndows 10 despite following exact instructions on the official site.

from moviepy import editor
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.0.8-Q16\convert.exe"})


FONT_TYPE    = 'Courier'
FONT_SIZE    = 40
FONT_COLOR   = 'yellow'
STROKE_COLOR = 'black'
STROKE_WIDTH = 2


def create_text_clip(text: str) -> editor.TextClip:
	return editor.TextClip(text, fontsize=FONT_SIZE, color=FONT_COLOR, stroke_color=STROKE_COLOR, stroke_width=STROKE_WIDTH, font=FONT_TYPE)


def annotate(clip, text: str):
	sub = create_text_clip(text)
	cvc = editor.CompositeVideoClip([clip, sub.set_pos(('center', 'bottom'))])

	return cvc.set_duration(clip.duration)


def pairwise(it: list):
	it = iter(it)
	while True:
		yield next(it), next(it)


def split_text_per_frame(text: str, video_frames_len: int, frame_rate: float) -> [(int, int, str)]:
	subs = text.split()
	inc  = max(video_frames_len / (len(subs) + 1), 0.01)

	l = range(0, video_frames_len)
	marks = []

	for i, e in zip(l,l[1:]):
		sub = " ".join(subs[:int(i / inc)])
		marks.append((i / frame_rate, e / frame_rate, sub))

	return marks

video_clip = editor.VideoFileClip('D:/GRID/s34/sbwe6a.mpg')

video_num_frames = sum((1 for _ in video_clip.iter_frames()))
video_frame_rate = video_clip.fps

print(video_num_frames)
print(video_frame_rate)
print()

split_subs      = split_text_per_frame('set blue with e six again', video_num_frames, video_frame_rate)
for s in split_subs: print(s)

annotated_clips = [annotate(video_clip.subclip(s[0], s[1]), s[2]) for s in split_subs]
final_clip      = editor.concatenate_videoclips(annotated_clips)

final_clip.write_videofile('misc/output.mpg')
