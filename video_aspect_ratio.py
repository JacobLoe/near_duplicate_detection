# adopted from https://stackoverflow.com/a/56254114
import subprocess
import json
def get_aspect_ratios(video_file):
    cmd = 'ffprobe -i "{}" -v quiet -print_format json -show_format -show_streams'.format(video_file)
    jsonstr = subprocess.check_output(cmd, shell=True, encoding='utf-8')
    r = json.loads(jsonstr)
    # look for "codec_type": "video". take the 1st one if there are multiple
    video_stream_info = [x for x in r['streams'] if x['codec_type']=='video'][0]
    if 'display_aspect_ratio' in video_stream_info and video_stream_info['display_aspect_ratio']!="0:1":
        a,b = video_stream_info['display_aspect_ratio'].split(':')
        display_aspect_ratio = int(a)/int(b)
    else:
        # some video do not have the info of 'display_aspect_ratio'
        w,h = video_stream_info['width'], video_stream_info['height']
        display_aspect_ratio = int(w)/int(h)
        ## not sure if we should use this
        #cw,ch = video_stream_info['coded_width'], video_stream_info['coded_height']
    if 'sample_aspect_ratio' in video_stream_info and video_stream_info['sample_aspect_ratio']!="0:1":
        # sample_aspect_ratio is typically referred to as pixel aspect ratio (PAR)
        # some video do not have the info of 'sample_aspect_ratio'
        a,b = video_stream_info['sample_aspect_ratio'].split(':')
        pixel_aspect_ratio = int(a)/int(b)
    else:
        pixel_aspect_ratio = display_aspect_ratio
    # compute storage_aspect_ratio from DAR/PAR
    storage_aspect_ratio = display_aspect_ratio/pixel_aspect_ratio
    return display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio