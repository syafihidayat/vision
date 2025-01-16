#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace rs2;

// Global variables
Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorKNN();
int lowhuepadi = 164, sminpadi = 105, vminpadi = 0;
int upperhuepadi = 178, smaxpadi = 255, vmaxpadi = 255;
int minDistance = 500, maxDistance = 1500;
int threshold_value = 255;

// const int Wreal = 0.155;
// const int Dreal = 1.0;
// const int Wpixels_calib = 64;
// float scale_factor = Wreal / Wpixels_calib; 

pipeline pipe;
rs2::config configs;

void setupTrackbars() {
    namedWindow("Trackbars", WINDOW_AUTOSIZE);
    createTrackbar("Low Hue", "Trackbars", &lowhuepadi, 255);
    createTrackbar("High Hue", "Trackbars", &upperhuepadi, 255);
    createTrackbar("Min Distance", "Trackbars", &minDistance, 2000);
    createTrackbar("Max Distance", "Trackbars", &maxDistance, 2000);
    createTrackbar("Threshold", "Trackbars", &threshold_value, 255);
}

void processFrame(const Mat& frame, const Mat& depth) {
    // Convert frame to HSV
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Create mask based on HSV range
    Scalar lower(lowhuepadi, sminpadi, vminpadi);
    Scalar upper(upperhuepadi, smaxpadi, vmaxpadi);
    Mat mask;
    inRange(hsv, lower, upper, mask);

    // Create depth masks
    Mat object_mask = depth > minDistance;
    Mat object_mask2 = depth < maxDistance;

    // Combine masks
    Mat combined_mask;
    bitwise_and(mask, object_mask2, combined_mask);

    // Morphological operations
    erode(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));
    dilate(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));

    // Draw centerline
    int y_position = 240; // lebar frame Y 480
    int x_position = 320; //lebar frame X 640
    Point top(x_position, 0);
    Point bottom(x_position, frame.rows - 1);
    Point left(0,y_position);
    Point right(frame.cols - 1, y_position);

    line(frame, left, right, Scalar(0,0,0), 2);
    line(frame,top,bottom, Scalar(0,0,0), 2);

    // Display coordinates
    // putText(frame, "Top: (" + to_string(right.x) + "," + to_string(left.y) + ")",
    //         right + Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
    // putText(frame, "Bottom: (" + to_string(bottom.x) + "," + to_string(bottom.y) + ")",
    //         bottom - Point(50, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

    // Find contours
    vector<vector<Point>> contours;
    findContours(combined_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Draw contours and process them
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 200 && area < 10000) {

            Rect bounding_rect = boundingRect(contours[i]);
            rectangle(frame, bounding_rect, Scalar(0, 255, 0), 2);
            Point center(bounding_rect.x + bounding_rect.width / 2, bounding_rect.y + bounding_rect.height / 2);
            circle(frame, center, 3, Scalar(255, 255, 255), -1);

            ushort dist = depth.at<ushort>(center);
            // float distance = dist > 0 ? dist / 1000.0f : -1;
            cout << "Bounding Box Center: (" << center.x << ", " << center.y << ")\n";
            // cout << "Estimated Distance: " << distance << "m\n";

            // string distance_text = "Distance: " + (distance >= 0 ? to_string(distance) + "m" : "Invalid");
            // putText(frame, distance_text, center + Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }
    }

    // Display results
    imshow("Combined Mask", combined_mask);
    imshow("Frame", frame);
}

void padiDetection() {
    while (true) {
        frameset frames = pipe.wait_for_frames();
        frame color_frame = frames.get_color_frame();
        depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) {
            cerr << "Error: Frames not captured correctly!" << endl;
            continue;
        }

        const int width = depth_frame.as<video_frame>().get_width();
        const int height = depth_frame.as<video_frame>().get_height();

        Mat depth(Size(width, height), CV_16U, (void *)depth_frame.get_data(), Mat::AUTO_STEP);
        Mat frame(Size(640, 480), CV_8UC3, (void *)color_frame.get_data(), Mat::AUTO_STEP);

        processFrame(frame, depth);

        int keyVal = waitKey(1) & 0xFF;
        if (keyVal == 113) { // 'q' to quit
            break;
        }
    }
}

int main() {
    // Configure and start the RealSense pipeline
    configs.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    configs.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipe.start(configs);

    setupTrackbars();

    try {
        padiDetection();
    } catch (const rs2::error &e) {
        cerr << "RealSense error: " << e.what() << endl;
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
    }

    // Stop the pipeline and clean up
    pipe.stop();
    destroyAllWindows();
    return 0;
}
