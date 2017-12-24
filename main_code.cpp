#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cuda;

#define SURF_DESCRIPTOR	 1
#define ORB_DESCRIPTOR	2

int main(int argc, char* argv[])
{	

cout<<"argvc : "<<argc<<endl;
	if(argc != 3) 
	{
		cout << "Image Stiching\nUsage: " <<argv[0] <<"{[SURF][ORB]}{image folder path for ex : ./data1}\n" << endl;
		return -1;
	} 
	String input = argv[1];
	int feature_descriptor = 0;

	/* Based on the command line argument decide whether it is SURF or ORB*/
	if ((0 == input.compare("SURF")) || (0 == input.compare("surf")))
	{
		feature_descriptor = SURF_DESCRIPTOR;
		cout<<"feature_descriptor : SURF "<<endl;
	}
	else if ((0 == input.compare("ORB")) || (0 == input.compare("orb")))
	{
		feature_descriptor = ORB_DESCRIPTOR;
		cout<<"feature_descriptor : ORB "<<endl;
	}
	else
	{
		cout <<"Invalid Feature Descriptor, Please input SURF or surf or ORB or orb"<<endl;
		return -1;
	}

	vector<String> filenames;/* input images file names (for example 1.jpg, 2.jpg etc) to store in the vector for image stitching */	
	String folder = argv[2]; /* name of the folder in whichinputimages are present (plese keep this older in current directory)*/	
	vector<Mat> img;
	Mat image;	
	glob(folder, filenames); /* glob fetches all the image files from the "folder" path and stores in "flenames" vector*/


	/* All the images will be read and if it the reading is success, then ll imges ne-by-one in the loop will be stored in the " img" 		   vector*/
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		Mat src = imread(filenames[i], CV_LOAD_IMAGE_UNCHANGED);
		
		if (!src.data)
		{
			cerr << "Problem loading image!!!" << endl;
		}	

		img.push_back(src);
	}
	
	Mat image1, image2;
	/* image2 will always have the 1st (index:0) image of panromo, image1 will have the next image*/
	image2 = img[0];
	Size size(512, 512);
	for (int k = 0; k < img.size()-1; k++)
	{
		/* 

		STEP 1: for loop count 0 : image1 will have the 2nd(index:1) image, once the stitiching of 1st and 2nd image is done the result  will be stored in image2

		STEP 2: for loop count 1 : image1 will have the 3rd(index:2)image, once the stitching is one with reslutant image from STEP1, the result will be stored in image2

		STEP 3: This loop will run till we finish stiching all the images and the final result will be stored in image2.

		*/
		image1 = img[k+1];

		/* resizing for computational efficiency */
		resize(image1, image1, size);
		resize(image2, image2, size);

		/*Convert to gray to extract descriptors*/
		Mat gray_image1;
		Mat gray_image2;

		/* if input images are gray scale then store as it is else convert to gray scale */
		if (image1.channels() != 3 && image2.channels() != 3)
		{
			gray_image1 = image1;
			gray_image2 = image2;
		}
		else
		{
			cvtColor(image1, gray_image1, CV_RGB2GRAY);
			cvtColor(image2, gray_image2, CV_RGB2GRAY);
		}
		

		if (!gray_image1.data || !gray_image2.data)
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return -1;
		}		

		/*
			keypoints_object : keypoints of the current image
			keypoints_scene  : keypoints of the scene i.e the stiched inmage stores in image2
		*/
		std::vector< KeyPoint > keypoints_object, keypoints_scene;
		Mat descriptors_object, descriptors_scene;

		if (SURF_DESCRIPTOR == feature_descriptor)
		{
			int minHessian = 400;
			Ptr<SURF> detector = SURF::create();			
			detector->setHessianThreshold(minHessian);
			cout<<"SURF detector set"<<endl;
			//--Step 1 : Detect the keypoints using SURF Detector
			detector->detect(gray_image1, keypoints_object);
			detector->detect(gray_image2, keypoints_scene);

			//--Step 2 : Calculate Descriptors (feature vectors)
			detector->compute(gray_image1, keypoints_object, descriptors_object);
			detector->compute(gray_image2, keypoints_scene, descriptors_scene);
		}
		else  /* ORB */
		{
			Ptr<ORB> detector = ORB::create();
			cout<<"ORB detector set"<<endl;

			//--Step 1 : Detect the keypoints using ORB Detector
			detector->detect(gray_image1, keypoints_object);
			detector->detect(gray_image2, keypoints_scene);

			//--Step 2 : Calculate Descriptors (feature vectors)
			detector->compute(gray_image1, keypoints_object, descriptors_object);
			detector->compute(gray_image2, keypoints_scene, descriptors_scene);
		}


		descriptors_scene.convertTo(descriptors_scene, CV_32F);
		descriptors_object.convertTo(descriptors_object, CV_32F);


		//--Step 3 : Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match(descriptors_object, descriptors_scene, matches);

		double max_dist = 0;
		double min_dist = 100;

		//--Quick calculation of min-max distances between keypoints
		for (int i = 0; i < descriptors_object.rows; i++)
		{

			double dist = matches[i].distance;

			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		//--Use only "good" matches (i.e. whose distance is less than 3 times min_dist )
		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors_object.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		std::vector< Point2f > obj;
		std::vector< Point2f > scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//--Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}

		//Find the Homography Matrix 
		Mat H = findHomography(obj, scene, CV_RANSAC, 3);
		
		cv::Mat result;
		// Use the homography Matrix to warp the images 
		warpPerspective(image1, result, H, Size(image1.cols + image2.cols, image1.rows));
	
		Mat half;
		half = result(Rect(0, 0, image2.cols, image2.rows));

		image2.copyTo(half);
		/* To remove the black portion after stitching, and confine in a rectangular region*/

		// vector with all non-black point positions
		std::vector<cv::Point> nonBlackList;
		nonBlackList.reserve(result.rows*result.cols);

		// add all non-black points to the vector
		// there are more efficient ways to iterate through the image

		for (int j = 0; j<result.rows; j++)
			for (int i = 0; i<result.cols; i++)
			{
				// if not black: add to the list
				//if (result.at<Vec3b>(j, i) != Vec3b(0,0,0))    //For colour images
				
				if (image1.channels() != 3 && image2.channels() != 3)
				{
					if (result.at<uint8_t>(j, i) != 0)
					{
						nonBlackList.push_back(Point(i, j));
					}
					
				}
				else
				{
					if (result.at<Vec3b>(j, i) != Vec3b(0, 0, 0))
					{
						nonBlackList.push_back(Point(i, j));
					}
				}
				
			}
		// create bounding rect around those points
		Rect bb = cv::boundingRect(nonBlackList);
		image2 = result(bb);

		if (img.size() < 3)
			break;
		else
			image1 = img[k];
		
	}
	imshow("Panorama", image2);
	imwrite("./image.jpg", image2);
	waitKey();
	return 0;
}
