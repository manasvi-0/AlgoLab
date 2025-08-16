import cv2
import numpy as np
import streamlit as st
import random

def visualiseImage(image):
    print("received")
    st.write("Visualise Convolution")
    options = ["Edge Detection", "Noise Removal", "Visualise Layers of CNN"]
    selected_option = st.selectbox("Choose an option:", options)

    if image is not None:

        H,W,_ = np.shape(image)

        st.image(cv2.cvtColor(st.session_state.uploaded_image, cv2.COLOR_BGR2RGB),
                    caption="Uploaded Image", use_container_width=True)

        if selected_option == "Edge Detection":
            kernel = st.radio("Choose Kernel", ["Prewitt", "Roberts", "Sobel", "Laplacian of Gaussian"])
            size = st.number_input("Enter an integer:", min_value=1, max_value=min(H,W), value=3, step=2)

            edge = visualiseKernelsEdgeDetection(image, type= kernel, kernel_size=size)

            output_image = edge.visualiseOutput()
            st.image(output_image, use_container_width = True)

        if selected_option == "Noise Removal":
            noise = st.radio("Choose Noise", ["Gaussian", "Salt and Pepper"])

            kernel = st.radio("Choose Filter", ["Median", "Bilateral", "Gaussian", "Average"])
            size = st.number_input("Enter an integer:", min_value=1, max_value=min(H,W), value=3, step=2)

            if (size%2 == 2):
                st.error("Only odd values of kernel sizes are allowed")

            if noise == "Gaussian":
                noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
                image = cv2.add(image, noise)
                st.image(image, use_container_width = True)
            elif noise == "Salt and Pepper":
                print(np.shape(image))
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                count = random.randint(100, H*W)

                for i in range(count):
                    y_coord=random.randint(0, H - 1)
            
                    # Pick a random x coordinate
                    x_coord=random.randint(0, W- 1)
                    
                    # Color that pixel to white
                    image[y_coord][x_coord] = 255

                count = random.randint(100, H*W)

                for i in range(count):
                    y_coord=random.randint(0, H - 1)
            
                    # Pick a random x coordinate
                    x_coord=random.randint(0, W - 1)
                    
                    # Color that pixel to black
                    image[y_coord][x_coord] = 0

                st.image(image, use_container_width = True)

            smoothening = visualiseKernelsBlurImage(image, type=kernel.lower(), kernel_size=size)

            output_image = smoothening.visualiseOutput()
            st.image(output_image, use_container_width = True)


        # Center the Clear Image button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Clear Image"):
            st.session_state.uploaded_image = None
            st.rerun()


#Allows the user to visualise various edge detection kernels
class visualiseKernelsEdgeDetection():
    def __init__(self, image, kernel_size = 3, axis = "x", type = None):
        self.input_image = image
        self.kernel = np.ones((kernel_size, kernel_size))
        #determines whether we want to detect the edges along the horizontal or the vertical axis
        self.axis = axis

        if type == "Sobel":
            if self.axis == "x":
                self.kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            else:
                self.kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        elif type == "Prewitt":
            if self.axis == "x":
                self.kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            else:
                self.kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        
        elif type == "Roberts":
            if self.axis == "x":
                self.kernel = np.array([[0, 1], [-1, 0]])
            
            else:
                self.kernel = np.array([[1, 0], [0, -1]])

        elif type == "LoG":
            self.kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    def visualiseKernel(self):
        #scale the kernel so that it is the range of 0 to 255
        kernel_normalised = (self.kernel - self.kernel.min())/(self.kernel.max() - self.kernel.min())
        kernel_display = (kernel_normalised*255).astype(np.uint8)
        cv2.imshow("Kernel to Display", kernel_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualiseOutput(self):
        output_image = cv2.filter2D(self.input_image, -1, self.kernel)
        # cv2.imshow("Output Image", output_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return output_image

class visualiseKernelsBlurImage():
    def __init__(self, image, kernel_size = 3, type = None):
        self.input_image = image

        if type == "median":
            self.filtered_image = cv2.medianBlur(self.input_image, kernel_size)

        elif type == "bilateral":
            self.filtered_image = cv2.bilateralFilter(self.input_image, kernel_size, 75, 75)

        elif type == "gaussian":
            self.filtered_image = cv2.GaussianBlur(self.input_image, (kernel_size, kernel_size), 0)

        else:
            kernel = np.ones((kernel_size, kernel_size))*1/kernel_size
            self.filtered_image = cv2.filter2D(self.input_image, -1, kernel)

    def visualiseOutput(self):
        return self.filtered_image

