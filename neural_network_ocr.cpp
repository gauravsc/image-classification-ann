#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "BOWToyLib.h"
#include <string>


using namespace cv;
using namespace std;


#define use_grey_image 1
#define use_red_image 0
#define use_blue_image 0
#define use_green_image 0
#define subtract_mean_mat 0

//Global variables 
const int threshold_value=30;
const int train_sample_count=30000;
const int test_sample_count=100;
const int thresh=100;
const int max_BINARY_value=255;


//Global enum to map categories with numbers
Mat train_data = Mat::zeros(train_sample_count, 784, CV_32F);
Mat train_classes = Mat::zeros(train_sample_count, 10, CV_32F);
Mat test_data = Mat::zeros(test_sample_count, 784, CV_32F);
Mat test_classes = Mat::zeros(test_sample_count, 10, CV_32F);
Mat pred_classes = Mat::zeros(test_sample_count, 10, CV_32F);
Mat mean_mat = Mat::zeros(28, 28, CV_32F);




// Global mean_mat Mat elements

/**
 * [get_vector Reshapes the 2 dimensional matrix of an image into 1 dimensional vector for passing into the learn algorithm]
 * @param  image
 * @param  image_as_vector
 * @return
 */
 Mat get_vector(Mat &image){

   return image.reshape(0,784);
 }

// /**
//  * [get_foreground Calculates the foreground from the given image using grabCut function of openCV]
//  * @param  image
//  * @return
//  */
// Mat get_foreground(Mat image){
//   Mat result,bgModel,fgModel;
//   Rect rectangle(32,32,32,32);
//   grabCut(image, result, rectangle, bgModel, fgModel, 1, GC_INIT_WITH_RECT); 
//   compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
//   Mat foreground(image.size(),CV_8UC3,cv::Scalar(0,0,0));
//   image.copyTo(foreground,result);
//   return Mat(foreground, rectangle);
// }



 Mat preprocess_image(Mat original_image){
  Mat bin_image;
  threshold(original_image, bin_image, 5, 1,0);
  //bin_image.convertTo(bin_image,CV_32F);
  return (bin_image);
}



void prepare_train_data(){
  std::ifstream  data("train_ocr.csv");
  int c;
  Mat image,vector_image,image_1,image_2;
  std::string line;
  int count=0,train_samples=0;
  std::getline(data,line);
  int row_index=0;
  cout<<"inside prepare train data function"<<endl;
  while(std::getline(data,line) && row_index<train_sample_count)
  {
    std::stringstream  lineStream(line);
    std::string        cell;
    count=0;
    int col_index=0;
    image= Mat::zeros(1,784, CV_8UC1);
    while(std::getline(lineStream,cell,',') )
    { 
      if(count==0){
        count=1;
        c=atoi(cell.c_str());
        train_classes.at<float>(row_index,c)=1;
      }else{
        c=atoi(cell.c_str());
        image.at<unsigned char>(0,col_index)=c;
      }
      col_index++;
    }
    cout<<row_index<<endl;
    //cout<<image<<endl;
    image_1=image.clone().reshape(0,28);
    image_1=preprocess_image(image_1);
    //namedWindow("image",1);
    //imshow("image",image_1);
    //waitKey(0);
    image_2=image_1.clone().reshape(0,1);
    //cout<<image_2;
    image_2.row(0).copyTo(train_data.row(row_index));
    row_index++;     

  }
  return ;
}




void calculate_mean(){
  std::ifstream  data("train_ocr.csv");
  int c;
  Mat image,vector_image,image_1,image_2;
  std::string line;
  int count=0,train_samples=0;
  std::getline(data,line);
  int row_index=0;
  cout<<"inside calculate mean_mat function"<<endl;
  while(std::getline(data,line) && row_index<train_sample_count)
  {
    std::stringstream  lineStream(line);
    std::string        cell;
    count=0;
    int col_index=0;
    image= Mat::zeros(1,784, CV_8UC1);
    while(std::getline(lineStream,cell,',') )
    { 
      if(count==0){
        count=1;
       
      }else{
        c=atoi(cell.c_str());
        image.at<unsigned char>(0,col_index)=c;
      }
      col_index++;
    }
    cout<<row_index<<endl;
    image_1=image.clone().reshape(0,28);
    image_1.convertTo(image_1,CV_32F);
    mean_mat=mean_mat+image_1;
    row_index++;
  }
  Mat bin_image;
  mean_mat=mean_mat/train_sample_count;
  threshold(mean_mat, bin_image, 5, 1,0);
  mean_mat =Mat(bin_image);
  return ;
}


vector<int> get_predicted_classes(Mat mat_classification){
  vector<int> c;
  for(int i=0;i<mat_classification.rows;i++){
    int max=-10000,ind=-1;
    for(int j=0;j<mat_classification.cols;j++){
      if(mat_classification.at<float>(i,j)>max){
              max=mat_classification.at<float>(i,j);
              ind=j;
      }
      

    }
    c.push_back(ind);
  }
return c;
}


void prepare_test_data(){
 std::ifstream  data("train_ocr.csv");
  int c;
  Mat image,vector_image,image_1,image_2;
  std::string line;
  int count=0;
  std::getline(data,line);
  int row_index=0,test_count=0;
  cout<<"inside prepare test data function"<<endl;
  while(std::getline(data,line) && test_count<train_sample_count+test_sample_count)
  {

    if(test_count<train_sample_count){
      test_count++;
      continue;
    }

    test_count++;
    std::stringstream  lineStream(line);
    std::string        cell;
    count=0;
    int col_index=0;
    image= Mat::zeros(1,784, CV_8UC1);
    while(std::getline(lineStream,cell,',') )
    { 
      if(count==0){
        count=1;
        c=atoi(cell.c_str());
        test_classes.at<float>(row_index,c)=1;
      }else{
        c=atoi(cell.c_str());
        image.at<unsigned char>(0,col_index)=c;
      }
      col_index++;
    }

    image_1=image.clone().reshape(0,28);
    image_1=preprocess_image(image_1);
    image_2=image_1.clone().reshape(0,1);
    cout<<row_index<<endl;
    image_2.row(0).copyTo(test_data.row(row_index));
    row_index++;    
    } 
 return ;
}


float get_accuracy(){
  vector<int> actual_classes=get_predicted_classes(test_classes);
  vector<int> predict_classes=get_predicted_classes(pred_classes);
  for(int i=0;i<actual_classes.size();i++){
    cout<<actual_classes[i]<<"  "<<predict_classes[i]<<endl;
  }
  float sum=0;
  for(int i=0;i<actual_classes.size();i++){
    if(actual_classes[i]==predict_classes[i]){
      sum++;
    }

  }

return sum/test_sample_count;

}

int main( int argc, char** argv ){
  Mat neural_layers = Mat(4, 1, CV_32SC1);
  CvANN_MLP classifier;
  neural_layers.at<int>(0,0)=784;
  neural_layers.at<int>(1,0)=1500;
  neural_layers.at<int>(2,0)=1500;
  neural_layers.at<int>(3,0)=10;
  Mat sample_wts = Mat::ones(1,train_sample_count,CV_32FC1);
  //calculate_mean();
  prepare_train_data();
  classifier.create(neural_layers,CvANN_MLP::SIGMOID_SYM, 1, 1);
  //cout<<train_data;
  //cout<<train_classes;
  classifier.train(train_data,
   train_classes,
   sample_wts
   );
  //calculate_mean();
  prepare_test_data();
  classifier.predict(test_data,pred_classes);
  cout<<"Acuracy of the neural network is: "<<get_accuracy();



  return 0;
}