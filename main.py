import cv2
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def buildGauss(frame, levels):
    """
    Build Gaussian Pyramid
    :param frame: the frame to build the pyramid from
    :param levels: the number of levels in the pyramid
    :return: the Gaussian pyramid
    """
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


def reconstructFrame(pyramid, index, levels):
    """
    Reconstruct Frame
    :param pyramid: the pyramid to reconstruct the frame from
    :param index: the index of the pyramid
    :param levels: the number of levels in the pyramid
    :return: the reconstructed frame
    """
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


def bandpassFilter(videoFrameRate, minFrequency, maxFrequency, bufferSize):
    """
    Bandpass Filter
    :param videoFrameRate: the frame rate of the video
    :param minFrequency: the minimum frequency
    :param maxFrequency: the maximum frequency
    :param bufferSize: the size of the buffer
    :return: the frequencies and the mask
    """
    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)
    return frequencies, mask


def update_plotly_graph(fig, heart_rate_values, max_heart_rate=200):
    """
    Update Plotly graph with heart rate values
    :param fig: the Plotly figure
    :param heart_rate_values: the heart rate values
    :param max_heart_rate: the maximum heart rate
    :return: the updated Plotly figure
    """
    num_values = len(heart_rate_values)

    # Clear existing traces
    fig.data = []

    # Create new trace for heart rate graph
    x_vals = np.arange(num_values)
    y_vals = heart_rate_values

    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Heart Rate', line=dict(color='blue', width=2)))

    # Update layout with grid and other parameters
    fig.update_layout(
        title='Heart Rate and ECG Monitor',
        xaxis_title='Time',
        yaxis_title='Heart Rate',
        showlegend=True,
        legend=dict(x=0, y=1.0),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    )

    return fig


if __name__ == '__main__':
    realWidth = 640
    realHeight = 480
    videoWidth = 160
    videoHeight = 120
    videoChannels = 3
    videoFrameRate = 15
    levels = 3
    alpha = 170
    minFrequency = 1.0
    maxFrequency = 2.0
    bufferSize = 150
    bufferIndex = 0
    cap = cv2.VideoCapture(0)

    cap.set(3, realWidth)
    cap.set(4, realHeight)

    # Initialize Gaussian Pyramid
    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros(bufferSize)

    # Heart Rate Calculation Variables
    heartRateCalculationFrequency = 10
    heartRateBufferIndex = 0
    heartRateBufferSize = 10
    heartRateBuffer = np.zeros((heartRateBufferSize))

    # Bandpass Filter
    frequencies, mask = bandpassFilter(videoFrameRate, minFrequency, maxFrequency, bufferSize)

    # Create Plotly figure
    fig = make_subplots(rows=1, cols=1, subplot_titles=('Heart Rate Monitor',))

    # Start video capture loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            face = frame[y:y + h, x:x + w]
            fh_x, fh_y, fh_w, fh_h = 0.5, 0.18, 0.25, 0.15
            subface = [int(x + w * fh_x - (w * fh_w / 2.0)),
                       int(y + h * fh_y - (h * fh_h / 2.0)),
                       int(w * fh_w),
                       int(h * fh_h)]
            detectionFrame = frame[subface[1]:subface[1] + subface[3], subface[0]:subface[0] + subface[2]]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))
            cv2.rectangle(frame, (subface[0], subface[1]), (subface[0] + subface[2], subface[1] + subface[3]),
                          color=(0, 255, 0), thickness=2)

            # Construct Gaussian Pyramid
            videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            # Bandpass Filter
            fourierTransform[~mask] = 0

            # Grab a Pulse
            if bufferIndex % heartRateCalculationFrequency == 0:
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                heart_rate = hz * 60
                heartRateBuffer[heartRateBufferIndex] = heart_rate
                heartRateBufferIndex = (heartRateBufferIndex + 1) % heartRateBufferSize

            bufferIndex = (bufferIndex + 1) % bufferSize

            heart_rate_value = heartRateBuffer.mean()

            if np.count_nonzero(heartRateBuffer) > 0:
                heart_rate = "{:.2f}".format(heart_rate_value)
                cv2.putText(frame, f'Heart Rate: {heart_rate_value}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Calculating Heart Rate...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update Plotly graph
            update_plotly_graph(fig, heartRateBuffer.tolist(), max_heart_rate=200)

            # Convert Plotly figure to image
            img_bytes = fig.to_image(format="png")

            # Display frame with Plotly graph
            cv2.imshow("Heart Rate Monitor", frame)
            cv2.imshow("Heart Rate Graph", cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
