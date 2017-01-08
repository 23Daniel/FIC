#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <arpa/inet.h>
#include <iostream>
//#include <opencv2\highgui.h>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2\cv.h>
#include "opencv2/opencv.hpp"
#define dist 5

using namespace std;
using namespace cv;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

int H_MINR = 100;
int H_MAXR = 180;
int S_MINR = 94;
int S_MAXR = 239;
int V_MINR = 198;
int V_MAXR = 256;

int H_MING = 60;
int H_MAXG = 76;
int S_MING = 30;
int S_MAXG = 219;
int V_MING = 170;
int V_MAXG = 250;

int H_MINB = 92;
int H_MAXB = 255;
int S_MINB = 135;
int S_MAXB = 255;
int V_MINB = 176;
int V_MAXB = 255;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const std::string windowName = "Original Image";
const std::string windowName1 = "HSV Image";
const std::string windowName2 = "Thresholded Image";
const std::string windowName3 = "After Morphological Operations";
const std::string trackbarWindowName = "Trackbars";

char buffer[256];
int sockfd;


void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}

void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
}

string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}

void createTrackbars() {
	//create window for trackbars


	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf(TrackbarName, "H_MIN", H_MIN);
	sprintf(TrackbarName, "H_MAX", H_MAX);
	sprintf(TrackbarName, "S_MIN", S_MIN);
	sprintf(TrackbarName, "S_MAX", S_MAX);
	sprintf(TrackbarName, "V_MIN", V_MIN);
	sprintf(TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25 > 0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25 < FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25 > 0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25 < FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
	//cout << "x,y: " << x << ", " << y;

}
void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area > MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				//cout << x << "," << y;
				drawObject(x, y, cameraFeed);

			}


		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

void error(const char *msg)
{
    perror(msg);
    exit(0);
}
void client()
{
    int portno;
    char port[] = "20231";
    char hostname[] = "193.226.12.217";
    struct sockaddr_in serv_addr;
    struct hostent *server;
	
    portno = atoi(port);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    server = gethostbyname(hostname);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
        error("ERROR connecting");
}

void send_message(char* message)
{
    int n;
  	//bzero(buffer,256);
    //sprintf(buffer, "%s", message);
    n = write(sockfd,message,strlen(message));
    sleep(1);
    if (n < 0) 
         error("ERROR writing to socket");
    bzero(message,256);
}

int aprox(int a, int b)
{
	int x;
	x = a - b;
	
	if(x < 0)
	{
		x = - x;
	}
	if(x <= dist)
	{
		return 1;
	}
	return 0;
}

const char *drum(int x1, int y1, int x2, int y2, int x11, int y11)
{
	if(x1 < x2 && aprox(x1,x2) == 0)//ob1 mai st decat ob2
	{
		if(y1 > y11)//sus
		{
			return "rsfs";
		}
		
		if(x1 < x11)//dr
		{
			return "fs";
		}
		
		if(y1 < y11)//jos
		{
			return "lsfs";
		}
		
		if(x1 > x11)//st
		{
			return "rsrsfs";
		}
	}
	
	if(x1 > x2 && aprox(x1,x2) == 0)//ob1 mai dr decat ob2
	{
		if(y1 > y11)//sus
		{
			return "lsfs";
		}
		
		if(x1 < x11)//dr
		{
			return "lslsfs";
		}
		if(y1 < y11)//jos
		{
			return "rsfs";
		}
		if(x1 > x11)//st
		{
			return "fs";
		}
	}
	if(y1 < y2 && aprox(y1,y2) == 0)//ob1 mai sus decat ob2
	{
		if(y1 > y11)//sus
		{
			return "rsrsfs";
		}
		if(x1 < x11)//dr
		{
			return "rsfs";
		}
		if(y1 < y11)//jos
		{
			return "fs";
		}
		if(x1 > x11)//st
		{
			return "lsfs";
		}
	}
	if(y1 > y2 && aprox(y1,y2) == 0)//ob1 mai jos decat ob2
	{
		if(y1 > y11)//sus
		{
			return "fs";
		}
		if(x1 < x11)//dr
		{
			return "lsfs";
		}
		if(y1 < y11)//jos
		{
			return "lslsfs";
		}
		if(x1 > x11)//st
		{
			return "rsfs";
		}
	}
	return "";
}

int main(int argc, char* argv[])
{

	//some boolean variables for different functionality within this
	//program
	bool trackObjects = true;
	bool useMorphOps = true;

	Point pR;
	Point pB;
	Point pG;
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat thresholdR;
	Mat thresholdB;
	Mat thresholdG;
	//x and y values for the location of the object
	int xR = 0, yR = 0;
	int xB = 0, yB = 0;
	int xG = 0, yG = 0;
	//create slider bars for HSV filtering
	createTrackbars();
	//video capture object to acquire webcam feed
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)
	capture.open("rtmp:://172.16.254.63/live/live");
	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop

	client();
	/*sprintf(buffer, "%s", "l");
	send_message(buffer);
	sprintf(buffer, "%s", "s");
	send_message(buffer);
	close(sockfd);*/
	
	while (1) {


		//store image to matrix
		capture.read(cameraFeed);
		//convert frame from BGR to HSV colorspace
		if (cameraFeed.empty() != 0) {
			return 1;
		}
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		//filter HSV image between values and store filtered image to
		//threshold matrix
		inRange(HSV, Scalar(H_MINR, S_MINR, V_MINR), Scalar(H_MAXR, S_MAXR, V_MAXR), thresholdR);
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if (useMorphOps)
			morphOps(thresholdR);
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		if (trackObjects)
			trackFilteredObject(xR, yR, thresholdR, cameraFeed);
      
		inRange(HSV, Scalar(H_MINB, S_MINB, V_MINB), Scalar(H_MAXB, S_MAXB, V_MAXB), thresholdB);
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if (useMorphOps)
			morphOps(thresholdB);
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		if (trackObjects)
			trackFilteredObject(xB, yB, thresholdB, cameraFeed);
		
		inRange(HSV, Scalar(H_MING, S_MING, V_MING), Scalar(H_MAXG, S_MAXG, V_MAXG), thresholdG);
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if (useMorphOps)
			morphOps(thresholdG);
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		if (trackObjects)
			trackFilteredObject(xG, yG, thresholdG, cameraFeed);
		
		//Client Socket part
		sprintf(buffer, "%s", drum(xR, yR, xB, yB, xG, yG);
		send_message(buffer);
		bzero(buffer,256);
		
		//show frames
		imshow(windowName2, thresholdR);
		imshow(windowName2, thresholdB);
		imshow(windowName2, thresholdG);
		imshow(windowName, cameraFeed);
		//imshow(windowName1, HSV);
		setMouseCallback("Original Image", on_mouse, &pR);
		setMouseCallback("Original Image", on_mouse, &pB);
		setMouseCallback("Original Image", on_mouse, &pG);
		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
		waitKey(30);
		
	}
	close(sockfd);
	return 0;
}
