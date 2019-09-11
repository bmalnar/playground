// #include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/imgproc.hpp"
#include "opencv2/core/highgui.hpp"

#include <iostream>
#include <iomanip>
#include "stdio.h"

#include <sys/stat.h>

using namespace std;
using namespace cv;

/*
A Mapping of Type to Numbers in OpenCV

C1	C2	C3	C4
CV_8U	0	8	16	24
CV_8S	1	9	17	25
CV_16U	2	10	18	26
CV_16S	3	11	19	27
CV_32S	4	12	20	28
CV_32F	5	13	21	29
CV_64F	6	14	22	30
*/

void _PrintMatrix(const char *pMessage, cv::Mat &mat)
{
	printf("%s\n", pMessage);

	int rows_cnt = mat.rows > 2 ? 2 : mat.rows; 
	int cols_cnt = mat.cols; // > 2 ? 2 : mat.cols;

	for (int r = 0; r < rows_cnt; r++) {
		for (int c = 0; c < cols_cnt; c++) {

			switch (mat.depth())
			{
			case CV_8U:
			{
				printf("%*u ", 3, mat.at<uchar>(r, c));
				break;
			}
			case CV_8S:
			{
				printf("%*hhd ", 4, mat.at<schar>(r, c));
				break;
			}
			case CV_16U:
			{
				printf("%*hu ", 5, mat.at<ushort>(r, c));
				break;
			}
			case CV_16S:
			{
				printf("%*hd ", 6, mat.at<short>(r, c));
				break;
			}
			case CV_32S:
			{
				printf("%*d ", 6, mat.at<int>(r, c));
				break;
			}
			case CV_32F:
			{
				printf("%*.4f ", 10, mat.at<float>(r, c));
				break;
			}
			case CV_64F:
			{
				printf("%*.4f ", 10, mat.at<double>(r, c));
				break;
			}
			}
		} printf("\n");
	} printf("\n");
}

void print_mat(cv::Mat mat) {

	int height = mat.size().height;
	int width = mat.size().width;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cout << mat.at<int>(i, j) << " ";
		}
		cout << endl;
	}
}

void print_mat_rows(cv::Mat mat, int numrows, int prec) {

	// cout << mat.rowRange(0, 2);

	int height = mat.size().height;
	int width = mat.size().width;

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < width; j++) {
			//cout << mat.at<unsigned int>(i, j) << " ";
			// printf("%u ", mat.at<unsigned int>(i, j));
			printf("%*u ", 3, mat.at<uchar>(i,j));
		}
		cout << endl;
	}
}

float* polyfit(cv::vector<Point> poly_points, int poly_order) {

	int i, j, k;
	int num_points = poly_points.size();
	float* a = new float[poly_order + 1];
	// B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
	float** B = new float*[poly_order + 1];
	for (i = 0; i < poly_order + 1; i++) {
		B[i] = new float[poly_order + 2];
	}
	// Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
	float* X = new float[2 * poly_order + 1];
	//Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
	float* Y = new float[poly_order + 1];

	// Initialize arrays
	for (i = 0; i < 2 * poly_order + 1; i++) {
		X[i] = 0;
		for (j = 0; j < num_points; j++) {
			X[i] = X[i] + pow(poly_points.at(j).x, i);
		}
		// Consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
	}

	for (i = 0; i <= poly_order; i++) {
		for (j = 0; j <= poly_order; j++) {
			B[i][j] = X[i + j];
			//Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
		}
	}

	for (i = 0; i < poly_order + 1; i++) {
		Y[i] = 0;
		for (j = 0; j < num_points; j++) {
			// Y[i] = Y[i] + pow(x[j], i) * y[j];
			Y[i] = Y[i] + pow(poly_points.at(j).x, i) * poly_points.at(j).y;
		}
		//consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
	}

	//load the values of Y as the last column of B(Normal Matrix but augmented)
	for (i = 0; i <= poly_order; i++) {
		B[i][poly_order + 1] = Y[i];
	}

	//n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations
	poly_order = poly_order + 1;
	/*
cout << "The normal (augmented matrix):" << endl;
for (i = 0; i < N; i++) {
	for (j=0;j<=N;j++) {
		cout << B[i][j] << setw(16);
	}
	cout << endl;
}
*/

// From now Gaussian Elimination starts (can be ignored) to solve the set of linear equations (Pivotisation)
	for (i = 0; i < poly_order; i++) {
		for (k = i + 1; k < poly_order; k++) {
			if (B[i][i] < B[k][i]) {
				for (j = 0; j <= poly_order; j++) {
					float temp = B[i][j];
					B[i][j] = B[k][j];
					B[k][j] = temp;
				}
			}
		}
	}

	// Loop to perform the gauss elimination
	for (i = 0; i < poly_order - 1; i++) {
		for (k = i + 1; k < poly_order; k++) {
			float t = B[k][i] / B[i][i];
			for (j = 0; j <= poly_order; j++) {
				//make the elements below the pivot elements equal to zero or elimnate the variables
				B[k][j] = B[k][j] - t * B[i][j];
			}
		}
	}

	// back-substitution
	for (i = poly_order - 1; i >= 0; i--) {                        //x is an array whose values correspond to the values of x,y,z..
		a[i] = B[i][poly_order];                //make the variable to be calculated equal to the rhs of the last equation
		for (j = 0; j < poly_order; j++) {
			if (j != i) {            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
				a[i] = a[i] - B[i][j] * a[j];
			}
		}
		a[i] = a[i] / B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
	}

	cout << "a: " << endl;
	for (i = 0; i < poly_order; i++) {
		cout << "a[" << i << "] = " << a[i] << endl;
	}

	return a;
}
int exists(const char *name)
{
	struct stat   buffer;
	return (stat(name, &buffer) == 0);
}

cv::Mat window_mask(int width, int height, cv::Mat img_ref, int center, int level) {
	cv::Mat ret = cv::Mat::zeros(img_ref.size(), img_ref.type());
	int a = int(img_ref.size().height - (level + 1)*height);
	int b = int(img_ref.size().height - level * height);
	int c = std::max(0, (int)(center - width / 2));
	int d = std::min((int)(center + width / 2), img_ref.size().width);
	for (int i = a; i <= b; i++) {
		for (int j = c; j <= d; j++) {
			ret.at<float>(i, j) = 1;
		}
	}
	return ret;
}
cv::Mat pt_mask(int width, int height, cv::Mat img_ref, int center, int level) {
	cv::Mat ret = cv::Mat::zeros(img_ref.size(), img_ref.type());
	int a = int(img_ref.size().height - (level + 1)*height);
	int b = int(img_ref.size().height - level * height);
	int c = std::max(0, (int)(center - width / 2));
	int d = std::min((int)(center + width / 2), img_ref.size().width);
	for (int i = a; i <= b; i++) {
		for (int j = c; j <= d; j++) {
			ret.at<float>(i, j) = img_ref.at<float>(i, j);
		}
	}
	return ret;
}

cv::Mat conv(cv::Mat in, int conv_size) {

	cout << "conv " << in.size().width << " " << in.size().height << endl;
	cv::Mat conv = cv::Mat::zeros(in.size(), in.depth());

	return conv;
}

std::vector<Point> my(cv::Mat image, int window_width, int window_height) {

	cout << image.type() << endl;
	print_mat_rows(image, 2, 0);

	std::vector<Point> window_centroids;
	cv::Mat window = cv::Mat::zeros(window_width, 1, CV_32F);
	int a = (int)(image.size().height / 2);
	int b = (int)(image.size().width / 2);
	cv::Mat sub_image_l = image(cv::Range(a, image.size().height), cv::Range(0, b));
	cv::Mat sub_image_r = image(cv::Range(a, image.size().height), cv::Range(b, image.size().width));

	cout << sub_image_l.size().height << " " << sub_image_l.size().width << " " << sub_image_l.type() << endl;
	
	cv::Mat l_sum = cv::Mat::zeros(1, sub_image_l.size().width, CV_32F);
	cout << l_sum.size().height << " " << l_sum.size().width << " " << l_sum.type() << endl;

	cv::Mat r_sum;
	cv::reduce(sub_image_l, l_sum, 0, CV_REDUCE_SUM, CV_32F);
	cv::Mat convl = conv(l_sum, window_width);
	//print_mat_rows(sub_image_l, 2, 0);
	//print_mat_rows(l_sum, 2, 5);
	//cv::reduce(sub_image_r, r_sum, 0, CV_REDUCE_SUM);
	_PrintMatrix("l_sum", l_sum);
	return window_centroids;
}

float calc_window_weight(cv::Mat image, int i_start, int j_start, int window_width, int window_height) {

	float sum = 0.0f;
	for (int i = i_start; i < i_start + window_height && i < image.size().height; i++) {
		for (int j = j_start; j < j_start + window_width && j < image.size().width; j++) {
			sum += (float)image.at<uchar>(i, j);
		}
	}

	return sum;
}

void get_windows2(cv::vector<Point> &wins_l, cv::vector<Point> &wins_r, cv::Mat image, int window_width, int window_height) {

	// std::vector<Point> windows;

	int im_width = image.size().width;
	int im_height = image.size().height;

	int im_half_width = (int)(im_width / 2);

	cout << "get_windows: " << im_width << " " << im_height << " " << im_half_width << endl;

	for (int i = 0; i < im_height; i += window_height) {

		float left_max = 0, right_max = 0;
		Point left_point, right_point;

		for (int j = 0; j < im_width; j += window_width) {

			
			float sum = calc_window_weight(image, i, j, window_width, window_height);

			//cout << "get_windows iter: " << i << " " << j << " " << sum << endl;

			if (j < im_half_width && sum > left_max) {
				left_max = sum;
				left_point = Point(i, j);
				//cout << "Added left " << endl;
			} else if (j > im_half_width && sum > right_max) {
				right_max = sum;
				right_point = Point(i, j);
				//cout << "Added right " << endl;
			}
		}

		wins_l.push_back(left_point);
		wins_r.push_back(right_point);
	}

	for (std::vector<Point>::const_iterator i = wins_l.begin(); i != wins_l.end(); ++i) {
		std::cout << *i << ' ';
	}
	for (std::vector<Point>::const_iterator i = wins_r.begin(); i != wins_r.end(); ++i) {
		std::cout << *i << ' ';
	}
	cout << endl;
}

std::vector<Point> get_windows(cv::Mat image, int window_width, int window_height) {

	std::vector<Point> windows;

	int im_width = image.size().width;
	int im_height = image.size().height;

	int im_half_width = (int)(im_width / 2);

	cout << "get_windows: " << im_width << " " << im_height << " " << im_half_width << endl;

	for (int i = 0; i < im_height; i += window_height) {

		float left_max = 0, right_max = 0;
		Point left_point, right_point;

		for (int j = 0; j < im_width; j += window_width) {


			float sum = calc_window_weight(image, i, j, window_width, window_height);

			//cout << "get_windows iter: " << i << " " << j << " " << sum << endl;

			if (j < im_half_width && sum > left_max) {
				left_max = sum;
				left_point = Point(i, j);
				//cout << "Added left " << endl;
			}
			else if (j > im_half_width && sum > right_max) {
				right_max = sum;
				right_point = Point(i, j);
				//cout << "Added right " << endl;
			}
		}

		windows.push_back(left_point);
		windows.push_back(right_point);
	}

	for (std::vector<Point>::const_iterator i = windows.begin(); i != windows.end(); ++i) {
		std::cout << *i << ' ';
	}
	cout << endl;

	return windows;
}
/*
def myp(image, window_width, window_height, margin) :
	window_centroids = [] # Store the(left, right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions

	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# and then np.convolve the vertical image slice with the window template

	# Sum quarter bottom of image to get slice, could use a different ratio
	a = int(3 * image.shape[0] / 6)
	b = int(image.shape[1] / 2)

	ii = image[a:, : b]
	l_sum = np.sum(image[a:, : b], axis = 0)
	print("l_sum = " + str(l_sum))
	print type(l_sum)
	print l_sum.shape
	print type(ii)
	print ii.shape
	print ii

	convl = np.convolve(window, l_sum)
	l_center = np.argmax(convl) - window_width / 2
	c = int(3 * image.shape[0] / 6)
	d = int(image.shape[1] / 2)
	r_sum = np.sum(image[c:, d : ], axis = 0)
	convr = np.convolve(window, r_sum)
	r_center = np.argmax(convr) - window_width / 2 + int(image.shape[1] / 2)

	# Add what we found for the first layer
	# window_centroids.append((l_center, r_center))
	# print(window_centroids)
	# return window_centroids
	# Go through each layer looking for max pixel locations
	for level in range((int)(image.shape[0] / window_height)) :
		# convolve the window into the vertical slice of the image
		a = int(image.shape[0] - (level + 1)*window_height)
		b = int(image.shape[0] - level * window_height)
		m = int(image.shape[1] / 2)
		n = m
		# print(a, b, m, n)
		image_layer_l = np.sum(image[a:b, : m], axis = 0)
		image_layer_r = np.sum(image[a:b, n : ], axis = 0)
		conv_signal_l = np.convolve(window, image_layer_l)
		conv_signal_r = np.convolve(window, image_layer_r)
		# print(image_layer)
		'''
		# Find the best left centroid by using past left center as a reference
		# Use window_width / 2 as offset because convolution signal reference is at right side of window, not center of window
		offset = window_width / 2
		l_min_index = int(max(l_center + offset - margin, 0))
		l_max_index = int(min(l_center + offset + margin, image.shape[1]))
		l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
		'''
		l_center = np.argmax(conv_signal_l)
		'''
		# Find the best right centroid by using past right center as a reference
		r_min_index = int(max(r_center + offset - margin, 0))
		r_max_index = int(min(r_center + offset + margin, image.shape[1]))
		r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
		'''
		r_center = m + np.argmax(conv_signal_r)
		# Add what we found for that layer
		window_centroids.append((l_center, r_center))

		return window_centroids
		*/

cv::Mat get_lane_points_and_windows(cv::Mat warped) {

	int window_width = 30;
	int window_height = 40;
	int margin = 100;
	int mid = int(warped.size().width / 2);
	std::vector<Point> window_centroids = my(warped, window_width, window_height);
	return warped;
}

cv::Mat pipeline(cv::Mat in) {

	cv::Mat hls;
	cv::cvtColor(in, hls, CV_BGR2HLS);

	vector<Mat> channels;
	split(hls, channels);

	Mat l_channel = Mat(channels[1]);
	Mat s_channel = Mat(channels[2]);

	int scale = 1;
	int delta = 0;
	int ddepth = CV_64F;

	Mat display;
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y, grad;
	Mat abs_grad_x, abs_grad_y;

	Mat sobel_s_x, abs_sobel_s_x, scaled_sobel_s_x, sx_binary;

	Sobel(s_channel, sobel_s_x, ddepth, 1, 0, 9, scale, delta, BORDER_DEFAULT);
	//_PrintMatrix("sobel_s_x", sobel_s_x);
	cout << "sobel_s_x type " << sobel_s_x.type() << endl;
	//_PrintMatrix("sobel_s_x", sobel_s_x);
	// cv::convertScaleAbs(sobel_s_x / 8, scaled_sobel_s_x);
	cv::convertScaleAbs(sobel_s_x / 256, scaled_sobel_s_x);
	//_PrintMatrix("scaled_sobel_s_x", scaled_sobel_s_x);
	cv::threshold(scaled_sobel_s_x, sx_binary, 180, 1, THRESH_BINARY);
	//_PrintMatrix("sx_binary", sx_binary);

	//sx_binary *= 256;
	//return sx_binary;
	Mat sobel_l_x, abs_sobel_l_x, scaled_sobel_l_x, lx_binary;

	Sobel(l_channel, sobel_l_x, ddepth, 1, 0, 9, scale, delta, BORDER_DEFAULT);
	cout << "sobel_l_x type " << sobel_l_x.type() << endl;
	cv::convertScaleAbs(sobel_l_x / 128, scaled_sobel_l_x);
	cv::threshold(scaled_sobel_l_x, lx_binary, 180, 1, THRESH_BINARY);

	Mat combined, combined256;
	cv::bitwise_or(sx_binary, lx_binary, combined);
	//_PrintMatrix("combined", combined);
	cv::convertScaleAbs(256 * combined, combined256);
	//_PrintMatrix("combined256", combined256);
	//return combined256;
	Mat sobx, soby, sobx_abs, soby_abs, sobgrad, sobgrad_low, sobgrad_up, sobgrad_final;
	Sobel(s_channel, sobx, CV_8U, 1, 0);
	Sobel(s_channel, soby, CV_8U, 0, 1);
	cv::convertScaleAbs(sobx, sobx_abs);
	cv::convertScaleAbs(soby, soby_abs);
	sobx_abs /= 256;
	soby_abs /= 256;
	Mat sobx_abs_f, soby_abs_f;
	sobx_abs.convertTo(sobx_abs_f, CV_64F);
	soby_abs.convertTo(soby_abs_f, CV_64F);
	_PrintMatrix("sobx", sobx_abs_f);
	//_PrintMatrix("soby", soby_abs_f);
	cv::phase(soby_abs_f, sobx_abs_f, sobgrad);
	_PrintMatrix("sobgrad", sobgrad);
	cout << "sobgrad type " << sobgrad.type() << endl;
	Mat tempa;
	cv::convertScaleAbs(256 * sobgrad, tempa);
	_PrintMatrix("tempa", tempa);
	/* 
	Mat temp, sobgrad_temp;
	sobgrad_temp = 10 * sobgrad;
	sobgrad_temp.convertTo(temp, CV_8U);
	_PrintMatrix("temp", temp);

	sobgrad_low = Mat(temp.size(), temp.type());
	sobgrad_up = Mat(temp.size(), temp.type());
	cv::threshold(temp, sobgrad_low, 7, 1, THRESH_BINARY_INV);
	cv::threshold(temp, sobgrad_up,  13, 1, THRESH_BINARY);
	cv::bitwise_and(sobgrad_low, sobgrad_up, sobgrad_final);
	//_PrintMatrix("sobgrad_low", sobgrad_low);
	//_PrintMatrix("sobgrad_up", sobgrad_up);
	*/
	sobgrad_final = Mat::zeros(sobgrad.size(), CV_8U);
	for (int i = 0; i < sobgrad.rows; i++) {
		for (int j = 0; j < sobgrad.rows; j++) {
			//double val = tempa.at<double>(i, j);
			//if (val > 0.7 && val < 1.3) {
			unsigned short val = tempa.at<unsigned short>(i, j);
			//cout << val << endl;
			if (val > 0 && val < 256) {
				sobgrad_final.at<unsigned short>(i, j) = 255;
			}
		}
	}

	Mat combined_final, sobgrad_final_bin;
	cv::threshold(sobgrad_final, sobgrad_final_bin, 180, 1, THRESH_BINARY);
	

	cv::bitwise_and(combined, sobgrad_final_bin, combined_final);
	combined_final *= 256;
	/**/
	_PrintMatrix("sobgrad_final", sobgrad_final);
	_PrintMatrix("combined", combined);
	_PrintMatrix("sobgrad_final_bin", sobgrad_final_bin);
	
	_PrintMatrix("combined_final", combined_final);
	
	/*
	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Mat gray;
	cv::cvtColor(in, gray, CV_BGR2GRAY);
	Sobel(gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	*/

	cout << "combined256 type " << combined256.type() << endl;
	cout << "grad type " << grad.type() << endl;
	// _PrintMatrix("grad", grad);
	Mat ret;
	ret = sobel_s_x;
	sobel_s_x.convertTo(ret, CV_16U);
	//imshow("ret",  ret);
	return sobgrad_final;
}

int main() {

	cout << exists("C:\\Users\\elaabrm\\Downloads\\projectVideo.mp4") << endl;
	cout << exists("C:\\Users\\elaabrm\\Downloads\\test2.jpg") << endl;

	// Create a VideoCapture object and open the input file
	// If the input is the web camera, pass 0 instead of the video file name
	VideoCapture cap("C:\\Users\\elaabrm\\Downloads\\project_video.mp4");

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	int im_w = 1280;
	int im_h = 720;
	int offset = 0;

	// src = np.float32([[500, 470], [780, 470], [1080, 650], [200, 650]])
	// dst = np.float32([[offset, offset], [img_size[0] - offset, offset], [img_size[0] - offset, img_size[1] - offset], [offset, img_size[1] - offset]])

	cv::Point2f src_vertices[4];
	src_vertices[0] = Point(500, 470);
	src_vertices[1] = Point(780, 470);
	src_vertices[2] = Point(1080, 650);
	src_vertices[3] = Point(200, 650);

	Point2f dst_vertices[4];
	dst_vertices[0] = Point(offset, offset);
	dst_vertices[1] = Point(im_w - offset, offset);
	dst_vertices[2] = Point(im_w - offset, im_h - offset);
	dst_vertices[3] = Point(offset, im_h - offset);

	Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
	Mat M_inv = getPerspectiveTransform(dst_vertices, src_vertices);

	cout << M.size().width << " " << M.size().height << endl;

	cv::MatIterator_<double> _it = M.begin<double>();
	for (; _it != M.end<double>(); _it++) {
		std::cout << *_it << std::endl;
	}

	_it = M_inv.begin<double>();
	for (; _it != M_inv.end<double>(); _it++) {
		std::cout << *_it << std::endl;
	}

	Mat mtx = (Mat_<double>(3, 3) << 1.15529427e+03, 0.00000000e+00, 6.66607659e+02,
		0.00000000e+00, 1.15207332e+03, 3.90091939e+02,
		0.00000000e+00, 0.00000000e+00, 1.00000000e+00);

	float data_mtx[3][3] = { {1.15529427e+03, 0.00000000e+00, 6.66607659e+02}, 
	{0.00000000e+00, 1.15207332e+03, 3.90091939e+02},
	{0.00000000e+00, 0.00000000e+00, 1.00000000e+00}};

	cout << "data_mtx = " << endl;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cout << data_mtx[i][j] << " ";
		}
		cout << endl;
	}

	Mat dist = (Mat_<double>(1, 5) << -2.39859361e-01, -2.71783516e-02, -6.46830818e-04, 6.94852396e-06,
		-1.99626622e-02);

	float dist_data[1][5] = { {-2.39859361e-01, -2.71783516e-02, -6.46830818e-04, 6.94852396e-06,
  -1.99626622e-02} };

	Mat rvecs = (Mat_<double>(18, 3) << 0.035787551948181034, -0.028348151896162854, -0.007378974918370405,
		0.513996679859966, -0.21794547365011274, 0.028517926443001263,
		0.03898754413948115, 0.4608697588726643, 0.006653594168759568,
		0.03740836168240398, 0.646416206001606, 0.010071740668489808,
		-0.32859639766080706, 0.6607602336256122, -0.41385204129211306,
		0.05889852535854729, -0.5167433562843025, -0.005548615452524798,
		-0.019741198891410898, -0.48588513727041116, 0.018734536696541737,
		0.0444074376941085, -0.46153196532093976, -0.05765278177490741,
		0.22064650015513176, -0.06346752806692323, 0.0118612209721304,
		0.18341644801312093, -0.0537829650974517, 0.0010465079908685193,
		0.08703901789007709, 0.38387224900074973, 0.0552650478146734,
		0.6383093016973135, -0.046835568156110584, 0.016539812257782744,
		-0.018329207628249158, 0.38558628631223474, -0.0026292061697080096,
		-0.44759809723966415, -0.06353406445291397, -0.018895923146439928,
		0.01870198804396533, 0.02376942384937375, -0.005544627082789631,
		0.021805468832337162, 0.6379019593066746, 0.0098189167240878,
		0.03164108432199019, -0.7026552441256677, -0.019574069302427857,
		-0.19120416053128075, -0.7577570290248707, 0.12034016378482212);
	float rvecs_data[18][3][1] = {
{{0.035787551948181034}, {-0.028348151896162854}, {-0.007378974918370405}},
{{0.513996679859966}, {-0.21794547365011274}, {0.028517926443001263}},
{{0.03898754413948115}, {0.4608697588726643}, {0.006653594168759568}},
{{0.03740836168240398}, {0.646416206001606}, {0.010071740668489808}},
{{-0.32859639766080706}, {0.6607602336256122}, {-0.41385204129211306}},
{{0.05889852535854729}, {-0.5167433562843025}, {-0.005548615452524798}},
{{-0.019741198891410898}, {-0.48588513727041116}, {0.018734536696541737}},
{{0.0444074376941085}, {-0.46153196532093976}, {-0.05765278177490741}},
{{0.22064650015513176}, {-0.06346752806692323}, {0.0118612209721304}},
{{0.18341644801312093}, {-0.0537829650974517}, {0.0010465079908685193}},
{{0.08703901789007709}, {0.38387224900074973}, {0.0552650478146734}},
{{0.6383093016973135}, {-0.046835568156110584}, {0.016539812257782744}},
{{-0.018329207628249158}, {0.38558628631223474}, {-0.0026292061697080096}},
{{-0.44759809723966415}, {-0.06353406445291397}, {-0.018895923146439928}},
{{0.01870198804396533}, {0.02376942384937375}, {-0.005544627082789631}},
{{0.021805468832337162}, {0.6379019593066746}, {0.0098189167240878}},
{{0.03164108432199019}, {-0.7026552441256677}, {-0.019574069302427857}},
{{-0.19120416053128075}, {-0.7577570290248707}, {0.12034016378482212}}
	};

	Mat tvecs = (Mat_<double>(18, 3) << -4.2067286378099, -2.328629936019235, 8.47232083057268,
		-2.067685092163532, -0.7879872669861475, 19.567611503807015,
		-16.946055556743882, -3.5938566340336395, 32.05990964717467,
		-0.17025414224410967, -3.53965659758442, 21.868887300431254,
		-5.9974276626559, -1.6459428894601775, 26.67484512204059,
		5.424565938038441, -4.517672329603618, 20.771839229031322,
		4.53590711878514, -1.5351702738461908, 19.98339770287975,
		4.933989332115921, -5.1129319654725265, 19.766656192661184,
		-3.9387318830028994, -1.373807445554384, 17.020008257152067,
		-3.5939817975381216, -4.1938659618583625, 17.786212788038686,
		-12.994057137300373, -5.67049110377465, 23.75383163336914,
		-3.81251301900247, -1.625266177574172, 7.954322174278309,
		-13.383053166766626, -0.5651944353444776, 24.54782963443558,
		-4.37530895449841, -3.08670074363757, 10.714937034022677,
		-4.8980740171925765, -3.953987173470921, 30.458138456276636,
		-9.57606294909838, -3.3816867864259788, 32.140515619927314,
		0.7572101403479617, -2.975737620299634, 19.575910887878646,
		-0.8205981409160775, -4.64690401680416, 21.711578309633243);
	float tvecs_data[18][3][1] = {
{{-4.2067286378099}, {-2.328629936019235}, {8.47232083057268}},
{{-2.067685092163532}, {-0.7879872669861475}, {19.567611503807015}},
{{-16.946055556743882}, {-3.5938566340336395}, {32.05990964717467}},
{{-0.17025414224410967}, {-3.53965659758442}, {21.868887300431254}},
{{-5.9974276626559}, {-1.6459428894601775}, {26.67484512204059}},
{{5.424565938038441}, {-4.517672329603618}, {20.771839229031322}},
{{4.53590711878514}, {-1.5351702738461908}, {19.98339770287975}},
{{4.933989332115921}, {-5.1129319654725265}, {19.766656192661184}},
{{-3.9387318830028994}, {-1.373807445554384}, {17.020008257152067}},
{{-3.5939817975381216}, {-4.1938659618583625}, {17.786212788038686}},
{{-12.994057137300373}, {-5.67049110377465}, {23.75383163336914}},
{{-3.81251301900247}, {-1.625266177574172}, {7.954322174278309}},
{{-13.383053166766626}, {-0.5651944353444776}, {24.54782963443558}},
{{-4.37530895449841}, {-3.08670074363757}, {10.714937034022677}},
{{-4.8980740171925765}, {-3.953987173470921}, {30.458138456276636}},
{{-9.57606294909838}, {-3.3816867864259788}, {32.140515619927314}},
{{0.7572101403479617}, {-2.975737620299634}, {19.575910887878646}},
{{-0.8205981409160775}, {-4.64690401680416}, {21.711578309633243}}
	};

	while (1) {

		Mat frame, gray;
		// Capture frame-by-frame
		cap >> frame;
		bool brk = false;


		frame = cv::imread("C:\\Users\\elaabrm\\Downloads\\test2.jpg");
		brk = true;

		// If the frame is empty, break immediately
		if (frame.empty())
			break;

		// cv::cvtColor(frame, gray, CV_BGR2GRAY);

		/*
		cout << frame.type() << endl;
		cout << gray.type() << endl;
		print_mat_rows(frame, 0, 0);
		print_mat_rows(gray, 0, 0);
		return 0;
		*/

		cv::Mat pipeline_out = pipeline(frame);
		imshow("Frame", pipeline_out);
		// Display the resulting frame
		//imshow("Frame", pipeline_out);

		// cout << grad_x.size().width << " " << grad_x.size().height << endl;

		Mat undist;
		cv::undistort(pipeline_out, undist, mtx, dist);
		Mat undist_warped;
		cv::warpPerspective(undist, undist_warped, M, undist.size(), INTER_LINEAR);
		imshow("undist warped",undist_warped);
		Mat undist_warped_bgr;
		cv::cvtColor(undist_warped, undist_warped_bgr, CV_GRAY2BGR);
		

		cout << "undist_warped.type() " << undist_warped.type() << endl;
		cout << "undist_warped_bgr.type() " << undist_warped_bgr.type() << endl;

		Mat white_img(undist_warped_bgr.size(), CV_8UC3, Scalar(255, 255, 255));

		

		// print_mat_rows(undist_warped, 2, 2);
		// _PrintMatrix("undist_warped", undist_warped);

		//cv::vector<Point> wins; 
		int win_size = 32;
		cv::vector<Point> wins_l, wins_r; 
		// cv::vector<Point> wins = get_windows2(wins_l, wins_r, undist_warped, win_size, win_size);
		get_windows2(wins_l, wins_r, undist_warped, win_size, win_size);
		cv::vector<Point> poly_points_l, poly_points_r;

		for (auto window = wins_l.begin(); window != wins_l.end(); window++) {
			Point p = *window;

			std::cout << p << std::endl;
			Rect r = Rect(p.y, p.x, win_size, win_size);
			rectangle(undist_warped_bgr, r, Scalar(0, 255, 0), 1, 8, 0);

			poly_points_l.push_back(p);
			/*
			for (int i = 0; i < win_size; i++) {
				for (int j = 0; j < win_size; j++) {
					poly_points_l.push_back(Point(p.y + i, p.x + j));
				}
			}
			*/
		}
		for (auto window = wins_r.begin(); window != wins_r.end(); window++) {
			Point p = *window;

			std::cout << p << std::endl;
			Rect r = Rect(p.y, p.x, win_size, win_size);
			rectangle(undist_warped_bgr, r, Scalar(0, 255, 0), 1, 8, 0);

			poly_points_r.push_back(p);
			/*
			for (int i = 0; i < win_size; i++) {
				for (int j = 0; j < win_size; j++) {
					poly_points_r.push_back(Point(p.y + i, p.x + j));
				}
			}
			*/
		}
		imshow("", undist_warped_bgr);
		cout << "Points left: " << poly_points_l.size() << endl;
		cout << "Points right: " << poly_points_r.size() << endl;
		float* coeff_l = polyfit(poly_points_l, 2);
		float* coeff_r = polyfit(poly_points_r, 2);

		for (auto point = poly_points_r.begin(); point != poly_points_r.end(); point++) {
			Point pt = *point;
			float ptx = pt.x;
			float pty_est = coeff_r[0] + coeff_r[1] * ptx + coeff_r[2] * ptx * ptx;
			Point pt_est = Point(ptx, pty_est);
			//std::cout << "Check r: " << pt << " " << pt_est << std::endl;
		}

		for (auto point = poly_points_l.begin(); point != poly_points_l.end(); point++) {
			Point pt = *point;
			float ptx = pt.x;
			float pty_est = coeff_l[0] + coeff_l[1] * ptx + coeff_l[2] * ptx * ptx;
			Point pt_est = Point(ptx, pty_est);
			//std::cout << "Check l: " << pt << " " << pt_est << std::endl;
		}

		const int im_height = undist_warped_bgr.size().height;
		Point* poly_draw_pts   = new Point[im_height * 2];
		Point* poly_draw_pts_l = new Point[im_height];
		Point* poly_draw_pts_r = new Point[im_height];
		int index = 0;
		for (int i = 0; i < im_height; i++, index++) {
			float x = (float)i;
			float y = coeff_l[0] + coeff_l[1] * x + coeff_l[2] * x * x;
			Point point_to_add = Point(y,x);
			//cout << "Add pt to draw L: " << i << " " << index << " " << point_to_add << endl;
			poly_draw_pts[index] = point_to_add;
			poly_draw_pts_l[i] = point_to_add;
		}

		for (int i = im_height - 1; i >= 0; i--, index++) {
			float x = (float)i;
			float y = coeff_r[0] + coeff_r[1] * x + coeff_r[2] * x * x;
			Point point_to_add = Point(y,x);
			//cout << "Add pt to draw R: " << i << " " << index << " " << point_to_add << endl;
			poly_draw_pts[index] = point_to_add;
			poly_draw_pts_r[im_height - i - 1] = point_to_add;
		}

		const Point *pts = (const cv::Point*) poly_draw_pts;
		const Point *pts_r = (const cv::Point*) poly_draw_pts_r;
		const Point *pts_l = (const cv::Point*) poly_draw_pts_l;
		polylines(undist_warped_bgr, &pts_r, &im_height, 1, true, Scalar(0, 0, 255), 15);
		polylines(undist_warped_bgr, &pts_l, &im_height, 1, true, Scalar(255, 0, 0), 15);
		fillConvexPoly(undist_warped_bgr, pts, index, Scalar(0, 255, 0));
		Mat combined;
		addWeighted(undist_warped_bgr, 1, white_img, 0.3, 0, combined);

		Mat combined_warped;
		cv::warpPerspective(undist_warped_bgr, combined_warped, M_inv, Size(1280, 720), INTER_LINEAR);

		Mat orig_combined;
		addWeighted(frame, 1, combined_warped, 0.3, 0, orig_combined);

		Mat aaa;
		pipeline_out.convertTo(aaa, CV_16U);
		imwrite("C:\\Users\\elaabrm\\Downloads\\test2out.jpg", aaa);
		
		imshow("final", orig_combined);
		//cv::Mat out = get_lane_points_and_windows(undist_warped);
		//if (brk) break;return 0;
		// Press  ESC on keyboard to exit
		
		char c = (char)waitKey(2500);
		if (c == 27)
			break;
	}

	// When everything done, release the video capture object
	// cap.release();
	
	// Closes all the frames
	cv::destroyAllWindows();

	return 0;
}

