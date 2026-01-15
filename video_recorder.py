import imageio

class VideoRecorder:
    def __init__(self, path, fps=10):
        self.writer = imageio.get_writer(path, fps=fps)

    def add_frame(self, frame):
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()
