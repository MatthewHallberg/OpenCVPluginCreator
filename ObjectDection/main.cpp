#include <opencv2/opencv.hpp>
#include <unistd.h>

using namespace cv;
using namespace std;

Mat frame;
Mat background;
Mat final_output;

void mouseEvent(int evt, int x, int y, int flags, void* param) {
    if (evt == EVENT_LBUTTONDOWN) {
        frame.copyTo(background);
        cout << "Background Changed!" << endl;
    }
}

int main(int argc, const char * argv[]) {
    
    namedWindow( "camWindow");
    setMouseCallback("camWindow", mouseEvent);
    
    //Check if we can get the webcam stream.
    VideoCapture capture(0);
    while(!capture.isOpened()) {
        cout << "Camera" << endl;
    }
    
    while (true) {
        capture.read(frame);
        resize(frame, frame, Size(frame.cols/2, frame.rows/2));

        // Laterally invert the image / flip the image
        flip(frame,frame,1);
        
        if (background.cols > 1){
            
            //Converting image from BGR to HSV color space.
            Mat hsv;
            cvtColor(frame, hsv, COLOR_BGR2HSV);

            Mat mask1,mask2;
            // Creating masks to detect the upper and lower bounds of color

            //red
            //            inRange(hsv, Scalar(0, 180, 70), Scalar(10, 255, 255), mask1);
            //            inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);

            //green
            inRange(hsv, Scalar(35, 40, 20), Scalar(100, 255, 255), mask1);
            inRange(hsv, Scalar(290, 100, 70), Scalar(300, 100, 100), mask2);

            // Generating the final mask
            mask1 = mask1 + mask2;

            Mat kernel = Mat::ones(3,3, CV_32F);
            morphologyEx(mask1,mask1,cv::MORPH_OPEN,kernel);
            morphologyEx(mask1,mask1,cv::MORPH_DILATE,kernel);

            // creating an inverted mask to segment out the cloth from the frame
            bitwise_not(mask1,mask2);
            Mat res1, res2, final_output;

            // Segmenting the cloth out of the frame using bitwise and with the inverted mask
            bitwise_and(frame,frame,res1,mask2);

            // creating image showing static background frame pixels only for the masked region
            bitwise_and(background,background,res2,mask1);

            // Generating the final augmented output.
            addWeighted(res1,1,res2,1,0,final_output);
            
            cout << "WTF" << endl;
            
            imshow("camWindow", final_output);
            
        } else {
            imshow("camWindow", frame);
        }
        
        if (waitKey(50) >= 0)
            break;
    }
    return 0;
}

