#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "BOWToyLib.h"

using namespace cv;
using namespace std;


#define use_grey_image 1
#define use_red_image 0
#define use_blue_image 0
#define use_green_image 1
#define subtract_mean 1


//Global enum to map categories with numbers
enum string_value { airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck};
std::map<std::string, string_value> map_string_values;


//Global variables 
const int threshold_value=30;
const int train_sample_count=3000;
const int test_sample_count=40;
const int thresh=100;
const int max_BINARY_value=255;


// Global mean Mat elements
Mat mean_mat; 

/**
 * [get_vector Reshapes the 2 dimensional matrix of an image into 1 dimensional vector for passing into the learn algorithm]
 * @param  image
 * @param  image_as_vector
 * @return
 */
 Mat get_vector(Mat image, Mat image_as_vector){
   int sz = image.cols*image.rows;
   image_as_vector=Mat(image.reshape(1,1024));
   return image_as_vector;
 }

/**
 * [get_foreground Calculates the foreground from the given image using grabCut function of openCV]
 * @param  image
 * @return
 */
 Mat get_foreground(Mat image){
  Mat result,bgModel,fgModel;
  Rect rectangle(32,32,32,32);
  grabCut(image, result, rectangle, bgModel, fgModel, 1, GC_INIT_WITH_RECT); 
  compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
  Mat foreground(image.size(),CV_8UC3,cv::Scalar(0,0,0));
  image.copyTo(foreground,result);
  return Mat(foreground, rectangle);
}

/**
 * [initialize_class_enums Initilize enum for mapping string names with indexes of classes in the labelled dataset]
 */
 void initialize_class_enums()
 {
  map_string_values["airplane"]=airplane;
  map_string_values["cat"]=cat;
  map_string_values["automobile"]=automobile;
  map_string_values["bird"]=bird;
  map_string_values["deer"]=deer;
  map_string_values["dog"]=dog;
  map_string_values["frog"]=frog;
  map_string_values["horse"]=horse;
  map_string_values["ship"]=ship;
  map_string_values["truck"]=truck;
  
}

/**
 * 
 */
 vector<int> get_classes(){
  std::ifstream  data("trainLabels.csv");
  vector<int> v ;
  int c;
  initialize_class_enums();
  std::string line;
  int count=0,train_samples=0;
  std::getline(data,line);
  while(std::getline(data,line) && train_samples<train_sample_count)
  {
    std::stringstream  lineStream(line);
    std::string        cell;
    count=0;
    while(std::getline(lineStream,cell,',') )
    { 
      if(count==0){
        count=1;
      }else{
        switch(map_string_values[cell]){
          case airplane: c=0;
          break;
          case automobile: c=1;
          break;
          case bird: c=2;
          break;
          case cat: c=3;
          break;
          case deer: c=4;
          break;
          case dog: c=5;
          break;
          case frog: c=6;
          break;
          case horse: c=7;
          break;
          case ship: c=8;
          break;
          case truck: c=9;
        }
        v.push_back(c);


      }

          // You have a cell!!!!
    }
    train_samples++;  
  }
  return (v);
}


/**
 * [get_border_extended Extend the border of the image using copyMakeBorder function of openCV]
 * @param  image
 * @return
 */
 Mat get_border_extended(Mat image){
  int top, bottom, left, right;
  Mat dst;
  Scalar value;
  top = (int) (image.rows); bottom = (int) (image.rows);
  left = (int) (image.cols); right = (int) (image.cols);
  dst=image;
  RNG rng(12345);
  value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
  copyMakeBorder( image, dst, top, bottom, left, right, BORDER_REPLICATE, value );
  // namedWindow("border extension",1);
  // imshow( "border extension", dst );
  // waitKey(0);
  return dst;
}


/**
 * [preprocess_image description]
 * @param  original_image
 * @return
 */
 Mat preprocess_image(Mat original_image){
  Mat gray_image,hist_equ_image,dst,dict,resize_hist_image,canny_output,channel[3],foreground,bin_image;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  HOGDescriptor hog;
  vector<float> ders;
  vector<Point>locs;
  int color;
  //namedWindow("original image",CV_AUTOSIZE);
  cvtColor(original_image,gray_image,CV_BGR2GRAY);
  equalizeHist(gray_image,hist_equ_image);
  //pyrDown(hist_equ_image, resize_hist_image, Size( hist_equ_image.cols/2, hist_equ_image.rows/2 ) ); 
  Canny(gray_image,canny_output,thresh,thresh+1,3);
  findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  Mat drawing = Mat::zeros(canny_output.size(), CV_8UC1);
  drawContours(drawing,contours,-1,255,CV_FILLED,8,hierarchy,4,Point());
  original_image=get_border_extended(original_image);
  //gray_image.convertTo(gray_image,CV_16SC1);
  foreground=get_foreground(original_image);
  //Mat resize_fg; 
  //cout<<foreground.rows;
 // pyrDown(foreground, resize_fg, Size( foreground.rows/3, foreground.cols/3) ); 
  split(foreground,channel);
  // namedWindow("original image",1);
  // imshow("original image",original_image);
  // namedWindow("foreground",1);
  // imshow("foreground",foreground);
  // imwrite("foreground_4.png",foreground);
  namedWindow("gray image",1);
  imshow("gray image",gray_image);
  imwrite("gray_5.png",gray_image);
  // namedWindow("blue image",1);
  // imshow("blue image",channel[0]);
  // namedWindow("green image",1);
  // imshow("green image",channel[1]);
  // namedWindow("red image",1);
  // imshow("red image",channel[2]);
  waitKey(0);

  if(use_grey_image==1){
    return gray_image;
  }
  
  if(use_red_image==1){
    color=2;
  } else if(use_green_image==1){
    color=1;
  }else{
    color=0;
  }
  threshold(channel[color], bin_image, 5, 1,0);
  return (bin_image);


}


/**
 * [get_difference_from_mean description]
 * @param  image
 * @return
 */
 Mat get_difference_from_mean(Mat image){
  Mat image1;
  //cout<<"inside mean difference calculator";
  image.convertTo(image1,CV_32SC1);
  return (image1-mean_mat);
}



/**
 * [get_mean_pixel_value description]
 * @return
 */
 Mat get_mean_pixel_value(){
  Mat original_image,gray_image,hist_equ_image,dst,dict,resize_hist_image,canny_output;
  Mat mean_mat,temp,prev_mean_mat;
  string image_name;
  int alpha=0.5,beta=0.5;
  std::vector<int> v=get_classes();
  for (int i=0;i<train_sample_count;i++){
    cout<<i<<endl;
    std::ostringstream s;
    s<<"./train/"<<i+1<<".png";
    image_name=s.str();
    original_image = imread(image_name);
    temp=preprocess_image(original_image);
    temp.convertTo(temp,CV_32SC1);
    //cout<<temp;
    if(i==0){
      mean_mat=Mat::zeros(temp.size(), CV_32SC1);
      mean_mat=mean_mat+temp;      
    }else{
      mean_mat=mean_mat+temp;
    }

  }
  return mean_mat/train_sample_count; 

}



/**
 * [start_testing description]
 * @param classifier
 */
 Mat start_testing(CvANN_MLP &classifier){
  string image_name;
  Mat original_image,gray_image,hist_equ_image,dst,dict,resize_hist_image,canny_output;
  Mat test_data = Mat::zeros(test_sample_count, 1024, CV_32F);
  Mat pred_class = Mat::zeros(test_sample_count, 10, CV_32F);
  Mat image_as_vector,drawing;
  
  for (int i=0;i<test_sample_count;i++){
    //cout<<i<<endl;
    std::ostringstream s;
    s<<"./train/"<<i+1<<".png";
    image_name=s.str();
    cout<<image_name<<endl;
    original_image = imread(image_name);
    cout<<"sending data for testing";

    drawing=preprocess_image(original_image);

    if(subtract_mean==1){
     drawing=Mat(get_difference_from_mean(drawing));
   }

   image_as_vector= get_vector(drawing,image_as_vector);
   image_as_vector=image_as_vector.t();  
   image_as_vector.row(0).copyTo(test_data.row(i));

 }

 classifier.predict(test_data,pred_class);
 return(pred_class);
}

/**
 * [get_train_data description]
 * @param  train_data
 * @param  train_classes
 * @return
 */
 Mat get_train_data(Mat train_data,Mat &train_classes) {
  Mat original_image,gray_image,hist_equ_image,dst,dict,resize_hist_image,canny_output;
  Mat image_as_vector,drawing;
  string image_name;
  std::vector<int> v=get_classes();
  cout<<"start collecting train data\n";
  for (int i=0;i<train_sample_count;i++){
    std::ostringstream s;
    s<<"./train/"<<i+1<<".png";
    image_name=s.str();
    cout<<image_name<<endl;
    original_image = imread(image_name);
    drawing=preprocess_image(original_image);
    drawing.convertTo(drawing,CV_16SC1);
    if(subtract_mean==1){
      drawing=Mat(get_difference_from_mean(drawing));
    }
    image_as_vector= get_vector(drawing,image_as_vector);
    image_as_vector=image_as_vector.t();  
    image_as_vector.row(0).copyTo(train_data.row(i));
    train_classes.at<float>(i,v[i])=1;
  }
  return train_data;
}
/**
 * [get_name_class Returns then name of the class]
 * @param  ind
 * @return
 */
 string get_name_class(int ind){
  switch(ind){
    case airplane: 
    return("airplane");
    break;
    case automobile: 
    return("automobile");
    break;
    case bird: 
    return("bird");
    break;
    case cat: 
    return("cat");
    break;
    case deer: 
    return("deer");
    break;
    case dog: 
    return("dog");
    break;
    case frog: 
    return("frog");
    break;
    case horse: 
    return("horse");
    break;
    case ship: 
    return("ship");
    break;
    case truck: 
    return("truck");
  }
}

/**
 * [print_predictions Print the predictions of classes]
 * @param pred_mat
 */
 float print_predictions(Mat pred_mat, Mat train_classes){

  cout<<"inside print prediction";
  vector<int> pred_ind;
  vector<int> actual_class;
  float sum;
  int max,ind;
  for (int i =0;i<pred_mat.rows;i++){
   max=-10000,ind=-1;
   for(int j=0;j<pred_mat.cols;j++){
    if(pred_mat.at<float>(i,j)>max){
      max=pred_mat.at<float>(i,j);
      ind=j;
    }
  }
  pred_ind.push_back(ind);
}
for (int k =0;k<train_classes.rows;k++){
 max=-10000;
 ind=-1;
 for(int l=0;l<train_classes.cols;l++){
  if(train_classes.at<float>(k,l)>max){
    max=train_classes.at<float>(k,l);
    ind=l;
  }
}
actual_class.push_back(ind);
}



for (int i=0;i<test_sample_count;i++){
  if(actual_class[i]==pred_ind[i])
    sum=sum+1;
}

for(int i=0;i<pred_ind.size();i++){
  cout<<"predicted: "<<get_name_class(pred_ind[i])<<" actual: "<<get_name_class(actual_class[i])<<endl;
}
return sum/test_sample_count;

}


/**
 * [main description]
 * @param  argc
 * @param  argv
 * @return
 */
 int main( int argc, char** argv ){
  Mat train_data = Mat::zeros(train_sample_count, 1024, CV_32F);
  Mat train_classes = Mat::zeros(train_sample_count, 10, CV_32F);
  Mat neural_layers = Mat(4, 1, CV_32SC1);
  CvANN_MLP classifier;
  neural_layers.at<int>(0,0)=1024;
  neural_layers.at<int>(1,0)=2000;
  neural_layers.at<int>(2,0)=2000;
  //neural_layers.at<int>(3,0)=2000;
  neural_layers.at<int>(3,0)=10;
  Mat sample_wts =  Mat::ones(1,train_sample_count,CV_32FC1);
  mean_mat=get_mean_pixel_value();
  Mat train_data_temp=get_train_data(train_data,train_classes);
  train_data_temp.convertTo(train_data,CV_32F);
  classifier.create(neural_layers,CvANN_MLP::SIGMOID_SYM, 1, 1);
  //cout<<train_classes;

  /**
   * classifier is supplied with the labelled image dataset along with the parameters for Neural Netowrk
   * It used backpropogation algorithm and uses both number of iterations and EPS as the criteria as target functions 
   */
   classifier.train(train_data,
     train_classes,
     sample_wts);

/*
Testing starts below
 /**
  * start testing inside the test function
  *
  */
  
  Mat pred_class=start_testing(classifier);
  cout<<"finished testing";
  cout<<"Accuracy is: "<<print_predictions(pred_class,train_classes);


  return 0;
}