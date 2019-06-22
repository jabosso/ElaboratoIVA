import argparse
from video import video_to_matrix

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required=True,help="Video")
args= vars(ap.parse_args())
print("{}".format(args["video"]))
video_to_matrix(args["video"])
