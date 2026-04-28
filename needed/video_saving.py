import matplotlib

def video_saving(p, filename):
    fps = 10
    matplotlib.use('Agg')
    # Use mpeg4 codec to avoid libopenh264 "Incorrect library version" errors
    matplotlib.rcParams['animation.codec'] = 'mpeg4'
    p.to_video(filename+".mp4", fps=fps)