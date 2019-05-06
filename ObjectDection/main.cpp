#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;
using namespace dnn;

// Initialize the parameters
float confThreshold = 0.2; // Confidence threshold
float nmsThreshold = .1;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image

vector<string> classes;
Mat blob;

void DrawDection(string label, Rect rect, Mat& frame){
    rectangle(frame, rect, Scalar(0, 0, 255),5);
    //Display the label at the top of the bounding box
    putText(frame, label, Point(rect.x, rect.y), FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255),3);
}

class Detection {
    public:
        string label;
        Rect2d box;
        Rect2d trackingBox;
        Ptr<Tracker> tracker;
    
        Detection(string objectName, Rect boundingBox,Mat& frame){
            label = objectName;
            box = boundingBox;
            trackingBox = box;
            tracker = TrackerKCF::create();
            tracker->init(frame, trackingBox);
        }
    
        void Draw(Mat& frame){
            DrawDection(label,box,frame);
        }
    
        void UpdateTracker(Mat& frame){
          tracker->update(frame, trackingBox);
            DrawDection(label,trackingBox,frame);
        }
};

vector<Detection> detections;

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame){
    //Draw a rectangle displaying the bounding box
    Rect rect(left, top, right - left, bottom - top);
    //Get the label for the class name and its confidence
    string label;// = format("%.2f", conf);
    if (!classes.empty()){
        CV_Assert(classId < (int)classes.size());
        label = classes[classId];// + ":" + label;
    }
    //add to current list
    Detection det = Detection(label,rect,frame);
    det.Draw(frame);
    detections.push_back(det);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net){
    static vector<String> names;
    if (names.empty()){
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs){
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i){
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold){
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i){
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

int main(int argc, const char * argv[]) {

    // Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3-tiny.cfg";
    String modelWeights = "yolov3-tiny.weights";
    
    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    //Capture stream from webcam.
    VideoCapture capture(0);
    
    //This variable will hold the image from the camera.
    Mat cameraFrame;
    //Read an image from the camera.
    capture.read(cameraFrame);
    
    //Check if we can get the webcam stream.
    if(!capture.isOpened()) {
        cout << "Could not open camera" << std::endl;
        return -1;
    }
    
    int framecount = 0;
    int modelInterval = 15;
    while (true) {
        
        //Read an image from the camera.
        capture.read(cameraFrame);
        
        if (framecount % modelInterval == 0){
            
            cout << "detecting: " << framecount << endl;
            
            //clear detections
            detections.clear();
            
            // Create a 4D blob from a frame.
            blobFromImage(cameraFrame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            
            //Sets the input to the network
            net.setInput(blob);
            
            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));
            
            // Remove the bounding boxes with low confidence
            postprocess(cameraFrame, outs);
        } else {
            //loop through all detections and track
            for(int i = 0; i < detections.size(); i++){
                detections[i].UpdateTracker(cameraFrame);
            }
        }
        
        framecount++;
        
        //make window half the size
        resize(cameraFrame, cameraFrame, Size(cameraFrame.cols/2, cameraFrame.rows/2));
        namedWindow( "Camera", WINDOW_AUTOSIZE);
        imshow("Camera", cameraFrame);
        
        if (waitKey(50) >= 0)
            break;
    }
    return 0;
}
