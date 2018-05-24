from preprocessing.extractor.detect_face_parts import extract_mouth

def video_to_frames_roi(input_loc, output_loc_frame, output_loc_roi):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    import time
    import cv2
    import os
    #try:
     #   os.mkdir(output_loc)
    #except OSError:
     #   pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    #print(cv2.CAP_PROP_FRAME_COUNT)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    #print ("Number of frames: ", video_length)
    count = 0
    #print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        #cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        output_frame = output_loc_frame + "/%#05d.jpg" % (count+1)
        output_roi = output_loc_roi + "/%#05d.jpg" % (count+1)
        print(output_frame)
        print(output_roi)
        cv2.imwrite(output_frame, frame)
        extract_mouth(output_frame, output_roi)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            #print ("Done extracting frames.\n%d frames extracted" % count)
            #print ("It took %d seconds forconversion." % (time_end-time_start))
            break

#video_to_frames("videos/s1/bbaf2n.mpg", "frames/s1/bbaf2n")